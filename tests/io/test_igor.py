import numpy as np
import xarray as xr
import xarray.testing

from erlab.io.igor import load_wave

WAVE0 = xr.DataArray(
    np.linspace(-0.3, 0.7, 6, dtype=np.float32),
    coords={"W": np.arange(6, dtype=np.float64)},
)


def test_backend(datadir):
    xarray.testing.assert_equal(xr.load_dataarray(datadir / "wave0.ibw"), WAVE0)
    xarray.testing.assert_equal(xr.load_dataset(datadir / "exp0.pxt")["wave0"], WAVE0)


def test_load_wave(datadir):
    xarray.testing.assert_equal(load_wave(datadir / "wave0.ibw"), WAVE0)
