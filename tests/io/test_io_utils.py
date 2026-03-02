import numpy as np
import pytest
import xarray as xr

from erlab.io.utils import save_as_hdf5


def test_save_as_hdf5_squeezes_size_one_dims_with_igor_compat(tmp_path) -> None:
    data = xr.DataArray(np.array([1.0]), dims=["x"], coords={"x": [0.0]})
    out = tmp_path / "single.h5"

    with pytest.warns(UserWarning, match="Dimensions with length 1 were squeezed"):
        save_as_hdf5(data, out, igor_compat=True)

    loaded = xr.load_dataarray(out, engine="h5netcdf")
    assert loaded.ndim == 0
    assert "IGORWaveScaling" in loaded.attrs
