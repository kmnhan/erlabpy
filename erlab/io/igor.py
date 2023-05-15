import os

import h5netcdf
import igor.igorpy
import numpy as np
import xarray as xr
from arpes import load_pxt

from erlab.io.utilities import find_first_file

__all__ = ["load_pxp", "load_h5", "load_ibw"]


def process_wave(arr):
    arr = arr.where(arr != 0)
    for d in arr.dims:
        arr = arr.sortby(d)
    return arr


def load_pxp(filename, recursive=False, silent=False, **kwargs):
    expt = load_pxt.read_experiment(filename, **kwargs)
    waves = dict()

    def unpack_folders(expt):
        for e in expt:
            try:
                arr = process_wave(load_pxt.wave_to_xarray(e))
                waves[arr.name] = arr
                if not silent:
                    print(arr.name)
            except AttributeError:
                pass
            if recursive and isinstance(e, igor.igorpy.Folder):
                unpack_folders(e)

    unpack_folders(expt)
    return waves


def load_h5(filename):
    ncf = h5netcdf.File(filename, mode="r", phony_dims="sort")
    ds = xr.open_dataset(xr.backends.H5NetCDFStore(ncf))
    for dv in ds.data_vars:
        wavescale = ds[dv].attrs["IGORWaveScaling"]
        ds = ds.assign_coords(
            {
                dim: wavescale[i + 1, 1]
                + wavescale[i + 1, 0] * np.arange(ds[dv].shape[i])
                for i, dim in enumerate(ds[dv].dims)
            }
        )
    return ds


def load_ibw(filename, data_dir=None):
    try:
        filename = find_first_file(filename, data_dir=data_dir)
    except (ValueError, TypeError):
        if data_dir is not None:
            filename = os.path.join(data_dir, filename)

    class ibwfile_wave(object):
        def __init__(self, fname):
            # self.wave = igor2.binarywave.load(fname)
            self.wave = load_pxt.read_single_ibw(fname)

    return load_pxt.wave_to_xarray(igor.igorpy.Wave(ibwfile_wave(filename)))
