import numpy as np
import xarray as xr
import xarray.testing

from erlab.io.igor import load_wave

WAVE0 = xr.DataArray(
    np.linspace(-0.3, 0.7, 6, dtype=np.float32),
    coords={"W": np.arange(6, dtype=np.float64)},
)
WAVE1 = xr.DataArray(
    np.linspace(-0.5, 1.5, 6, dtype=np.float32),
    coords={"W": np.arange(6, dtype=np.float64)},
)


def test_backend_dataarray(datadir) -> None:
    xarray.testing.assert_equal(xr.load_dataarray(datadir / "wave0.ibw"), WAVE0)


def test_backend_dataset(datadir) -> None:
    xarray.testing.assert_equal(xr.load_dataset(datadir / "exp0.pxt")["wave0"], WAVE0)


def test_backend_datatree(datadir) -> None:
    dt = xr.open_datatree(datadir / "exp1.pxt")
    assert dt.groups == ("/", "/wave0", "/testfolder", "/testfolder/wave1")

    xr.testing.assert_equal(dt["wave0/wave0"], WAVE0)
    xr.testing.assert_equal(dt["testfolder/wave1/wave1"], WAVE1)


def test_load_wave(datadir) -> None:
    xarray.testing.assert_equal(load_wave(datadir / "wave0.ibw"), WAVE0)
