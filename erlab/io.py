"""Functions for data IO.

PyARPES stores files using `pickle`, which can only be opened in the
environment it was saved in.
The function `save_as_netcdf` saves an array in the `netCDF4` format,
which can be opened in Igor as `hdf5` files. 

"""

import itertools
import os
import re
from pathlib import Path

import arpes
import arpes.endstations
import h5netcdf
import igor.igorpy

import numpy as np
import xarray as xr
from arpes import load_pxt
from astropy.io import fits

__all__ = [
    "showfitsinfo",
    "save_as_netcdf",
    "load_igor_pxp",
    "load_igor_ibw",
    "load_igor_h5",
    "load_ssrl",
]


def showfitsinfo(path: str):
    """Prints raw metadata from FITS file.

    Parameters
    ----------
    path : str
        Local path to `.fits` file.

    """
    with fits.open(path, ignore_missing_end=True) as hdul:
        hdul.verify("silentfix+warn")
        hdul.info()
        for i in range(len(hdul)):
            # print(f'\nColumns in {i:d}: {hdul[i].columns.names!r}')
            print(f"\nHeaders in {i:d}:\n{hdul[i].header!r}")


def fix_attr_format(da: xr.DataArray):
    """Discards attributes that are incompatible with the `netCDF4` file
    format.

    Parameters
    ----------
    da : xarray.DataArray
        Target array.

    Returns
    -------
    out : xarray.Dataset object
        Target array with incompatible attributes removed.

    """
    valid_dtypes = ["S1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]
    for key in da.attrs.keys():
        isValid = 0
        for dt in valid_dtypes:
            isValid += np.array(da.attrs[key]).dtype == np.dtype(dt)
        if not isValid:
            try:
                da = da.assign_attrs({key: str(da.attrs[key])})
            except:
                da = da.assign_attrs({key: ""})
    return da


def save_as_netcdf(data: xr.DataArray, filename: str, **kwargs):
    """Saves data in 'netCDF4' format.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray object to save.
    **kwargs : dict, optional
        Extra arguments to `DataArray.to_netcdf`: refer to the xarray
        documentation for a list of all possible arguments.

    """
    data = data.assign_attrs(provenance="")
    fix_attr_format(data).to_netcdf(
        filename,
        encoding={var: dict(zlib=True, complevel=5) for var in data.coords},
        **kwargs,
    )


def save_as_fits():
    # TODO
    pass


def parse_livepolar(wave, normalize=False):
    wave = wave.rename({"W": "eV", "X": "phi", "Y": "theta"})
    new_coords = {}
    new_coords["alpha"] = np.pi / 2
    new_coords["beta"] = np.deg2rad(wave.attrs["tilt"])
    new_coords["phi"] = np.deg2rad(wave["phi"])
    new_coords["theta"] = np.deg2rad(wave["theta"])
    new_coords["chi"] = np.deg2rad(wave.attrs["azimuth"])
    new_coords["hv"] = wave.attrs["hv"]
    new_coords["psi"] = 0.0
    new_coords["eV"] = wave["eV"] - wave.attrs["hv"]
    wave = wave.assign_coords(new_coords)
    wave = wave / wave.attrs["mesh_current"]
    if normalize:
        wave = arpes.preparation.normalize_dim(wave, "theta")
    return wave


def parse_livexy(wave):
    wave = wave.rename({"W": "eV", "X": "y", "Y": "x"})
    new_coords = {}
    new_coords["alpha"] = np.pi / 2
    new_coords["beta"] = np.deg2rad(wave.attrs["tilt"])
    # new_coords["phi"] = np.deg2rad(wave["phi"])
    new_coords["theta"] = np.deg2rad(wave.attrs["polar"])
    new_coords["chi"] = np.deg2rad(wave.attrs["azimuth"])
    new_coords["hv"] = wave.attrs["hv"]
    new_coords["psi"] = 0.0
    new_coords["eV"] = wave["eV"] - wave.attrs["hv"]
    wave = wave.assign_coords(new_coords)
    wave = wave / wave.attrs["mesh_current"]
    return wave


def process_wave(arr):
    arr = arr.where(arr != 0)
    for d in arr.dims:
        arr = arr.sortby(d)
    return arr


def load_igor_pxp(filename, recursive=False, silent=False, **kwargs):
    expt = load_pxt.read_experiment(filename, **kwargs)
    waves = dict()

    def unpack_folders(expt):
        for e in expt:
            try:
                arr = process_wave(load_pxt.wave_to_xarray(e))
                if "xy" in arr.name or "XY" in arr.name:
                    arr = parse_livexy(arr)
                elif "lp" in arr.name or "LP" in arr.name:
                    arr = parse_livepolar(arr)
                waves[arr.name] = arr
                if not silent:
                    print(arr.name)
            except AttributeError:
                pass
            if recursive and isinstance(e, igor.igorpy.Folder):
                unpack_folders(e)

    unpack_folders(expt)
    return waves


def load_igor_h5(filename):
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


def load_igor_ibw(filename, data_dir=None):
    try:
        filename = find_first_file(filename, data_dir=data_dir)
    except ValueError:
        pass

    class ibwfile_wave(object):
        def __init__(self, fname):
            self.wave = load_pxt.read_single_ibw(fname)

    return load_pxt.wave_to_xarray(igor.igorpy.Wave(ibwfile_wave(filename)))


def load_livexy(filename, data_dir=None):
    dat = load_igor_ibw(filename, data_dir)
    return parse_livexy(dat)


def load_ssrl(filename, data_dir=None, contains=None):
    try:
        filename = find_first_file(filename, data_dir=data_dir, contains=contains)
    except ValueError:
        pass

    ncf = h5netcdf.File(filename, mode="r", phony_dims="sort")
    attrs = dict(ncf.attrs)
    attr_keys_mapping = {
        "BL_energy": "hv",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "A": "chi",  # azi
        "T": "beta",  # polar
        "TA": "temperature_cryotip",  # cold head temp
        "TB": "temperature",  # sample temp
        "CreationTime": "creation_time",
        "HDF5Version": "HDF5_Version",
        "H5pyVersion": "h5py_version",
        "Notes": "description",
        "Sample": "sample",
        "User": "user",
        "Model": "analyzer_name",
        "SerialNumber": "serial_number",
        "Version": "version",
        "StartTime": "start_time",
        "StopTime": "stop_time",
        "UpdateTime": "update_time",
        "Duration": "duration",
        "LensModeName": "lens_mode",
        "CameraMode": "camera_mode",
        "AcquisitionTime": "acquisition_time",
        "WorkFunction": "sample_workfunction",
        "MeasurementMode": "acquisition_mode",
        "PassEnergy": "pass_energy",
        "MCP": "mcp_voltage",
        "BL_pexit": "exit_slit",
        "BL_I0": "photon_flux",  # 이거 맞나..?
        "BL_spear": "beam_current",  # 이거 맞나..?
        "YDeflection": "theta_DA",
    }
    coords_keys_mapping = {
        "Kinetic Energy": "eV",
        "ThetaX": "phi",
        "ThetaY": "theta",
    }
    fixed_attrs = {"analyzer_type": "hemispherical"}
    attr_to_coords = ["hv"]

    for k, v in fixed_attrs.items():
        attrs[k] = v

    for k, v in ncf.groups.items():
        ds = xr.open_dataset(xr.backends.H5NetCDFStore(v))
        if k == "Beamline":
            ds.attrs = {f"BL_{kk}": vv for kk, vv in ds.attrs.items()}

        attrs = attrs | ds.attrs
        if k == "Data":
            axes = [dict(v.groups[g].attrs) for g in v.groups]
            data = ds.rename_dims(
                {f"phony_dim_{i}": ax["Label"] for i, ax in enumerate(axes)}
            )
            for ax in axes:
                # try:
                #     mn, mx = ax["Minimum"], ax["Maximum"]
                # except KeyError:
                mn, mx = (
                    ax["Offset"],
                    ax["Offset"] + (ax["Count"] - 1) * ax["Delta"],
                )
                data = data.assign_coords(
                    {ax["Label"]: np.linspace(mn, mx, ax["Count"])}
                )
            if "Time" in data.variables:
                data = data.rename_vars(Count="spectrum", Time="time")
            else:
                data = data.rename_vars(Count="spectrum")

    for k in list(attrs.keys()):
        if k in attr_keys_mapping.keys():
            attrs[attr_keys_mapping[k]] = attrs.pop(k)

    data = data.rename({k: v for k, v in coords_keys_mapping.items() if k in data.dims})

    if "theta" not in itertools.product(attrs.keys(), data.dims):
        attrs["theta"] = 0.0

    attrs["alpha"] = 90.0
    attrs["psi"] = 0.0

    for a in ["alpha", "beta", "theta", "theta_DA", "chi", "phi", "psi"]:
        try:
            data = data.assign_coords({a: np.deg2rad(data[a])})
        except KeyError:
            try:
                data = data.assign_coords({a: np.deg2rad(attrs.pop(a))})
            except KeyError:
                continue

    for c in attr_to_coords:
        data = data.assign_coords({c: attrs.pop(c)})

    # data.attrs = attrs
    # data.spectrum.attrs = attrs
    if "time" in data.variables:
        out = data.spectrum / data.time
    else:
        out = data.spectrum
    out.attrs = attrs
    return out


def files_for_search(directory, contains=None):
    """Filters files in a directory for candidate scans.

    Here, this just means collecting the ones with extensions acceptable to the loader.
    """
    endbase = arpes.endstations.EndstationBase
    if contains is not None:
        return [
            f
            for f in os.listdir(directory)
            if os.path.splitext(f)[1] in endbase._TOLERATED_EXTENSIONS and contains in f
        ]
    return [
        f
        for f in os.listdir(directory)
        if os.path.splitext(f)[1] in endbase._TOLERATED_EXTENSIONS and "zap" not in f
    ]


def find_first_file(file, data_dir=None, contains=None, allow_soft_match=False):
    workspace = arpes.config.CONFIG["WORKSPACE"]
    if data_dir is None:
        data_dir = "data"
    try:
        workspace_path = os.path.join(workspace["path"], data_dir)
        workspace = workspace["name"]
    except KeyError:
        workspace_path = os.path.join(str(os.getcwd()), data_dir)
        workspace = "default"

    endbase = arpes.endstations.EndstationBase
    try:
        file = int(str(file))
    except ValueError:
        file = str(Path(file).absolute())
    scan_desc = {
        "file": file,
    }

    base_dir = workspace_path or os.path.join(arpes.config.DATA_PATH, workspace)
    dir_options = [
        os.path.join(base_dir, option) for option in endbase._SEARCH_DIRECTORIES
    ]
    patterns = [re.compile(m.format(file)) for m in endbase._SEARCH_PATTERNS]

    for dir in dir_options:
        try:
            files = files_for_search(dir, contains=contains)

            if endbase._USE_REGEX:
                for p in patterns:
                    for f in files:
                        m = p.match(os.path.splitext(f)[0])
                        if m is not None:
                            if m.string == os.path.splitext(f)[0]:
                                return os.path.join(dir, f)
            else:
                for f in files:
                    if os.path.splitext(file)[0] == os.path.splitext(f)[0]:
                        return os.path.join(dir, f)
                    if allow_soft_match:
                        matcher = os.path.splitext(f)[0].split("_")[-1]
                        try:
                            if int(matcher) == int(file):
                                return os.path.join(dir, f)  # soft match
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass

    if str(file) and str(file)[0] == "f":  # try trimming the f off
        return find_first_file(
            str(file)[1:], scan_desc, allow_soft_match=allow_soft_match
        )

    raise ValueError("Could not find file associated to {}".format(file))
