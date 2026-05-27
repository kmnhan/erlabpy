from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import erlab
import erlab.io.plugins.hers as hers

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
    if ip_session:
        ip_session.clear_instance()
    del start_ipython.already_called


def test_single_motor_scan_uses_nominal_axis(data_dir) -> None:
    loaded = erlab.io.load("20211216_00011.fits")

    np.testing.assert_allclose(loaded.beta, [-6.0, -5.5, -5.0])
    np.testing.assert_allclose(loaded.Alpha_readback, [-6.0, -5.49, -5.02])
    assert loaded.Alpha_readback.dims == ("beta",)


def test_extract_distortion_uses_determined_fit_degree(monkeypatch) -> None:
    alpha = np.linspace(16.0, 24.0, 11)
    eV = np.array([-5.0, -4.5, -4.0])
    alpha_motor = np.array([-1.0, 0.0, 1.0])
    data = xr.DataArray(
        np.ones((alpha.size, eV.size, alpha_motor.size)),
        dims=("alpha", "eV", "Alpha"),
        coords={"alpha": alpha, "eV": eV, "Alpha": alpha_motor},
    )

    def leading_edge(_data, *, dim, direction):
        assert dim == "alpha"
        assert direction == "positive"
        return xr.DataArray(
            [17.0, np.nan, 19.0], dims=("Alpha",), coords={"Alpha": alpha_motor}
        )

    original_polyfit = xr.DataArray.polyfit
    seen_degree = None

    def polyfit(self, *args, **kwargs):
        nonlocal seen_degree
        seen_degree = kwargs["deg"]
        return original_polyfit(self, *args, **kwargs)

    monkeypatch.setattr(erlab.analysis.interpolate, "leading_edge", leading_edge)
    monkeypatch.setattr(xr.DataArray, "polyfit", polyfit)

    hers._extract_distortion(data)

    assert seen_degree == 1
