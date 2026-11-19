from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab._provenance import FileVersion
from erlab.io.utils import save_as_hdf5


def test_save_as_hdf5_squeezes_size_one_dims_with_igor_compat(tmp_path) -> None:
    data = xr.DataArray(np.array([1.0]), dims=["x"], coords={"x": [0.0]})
    out = tmp_path / "single.h5"

    with pytest.warns(UserWarning, match="Dimensions with length 1 were squeezed"):
        save_as_hdf5(data, out, igor_compat=True)

    loaded = xr.load_dataarray(out, engine="h5netcdf")
    assert loaded.ndim == 0
    assert "IGORWaveScaling" in loaded.attrs


def test_file_replaced_during_versioned_read_closes_result(tmp_path) -> None:
    path = tmp_path / "scan.nc"
    original = xr.DataArray([1, 2], dims="x")
    original.to_netcdf(path, engine="h5netcdf")
    version = FileVersion.capture(path)
    changed = tmp_path / "next.nc"
    (original + 100).to_netcdf(changed, engine="h5netcdf")
    closed = []

    def read(filename):
        array = xr.load_dataarray(filename, engine="h5netcdf")
        array.set_close(lambda: closed.append(True))
        changed.replace(path)
        return array

    with pytest.raises(OSError, match=r"has changed.*checkpoint"):
        erlab.io.read_file_version(read=read, **asdict(version))
    assert closed == [True]
