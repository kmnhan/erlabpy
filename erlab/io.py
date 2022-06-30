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
import numpy as np
import xarray as xr
from astropy.io import fits

__all__ = ["showfitsinfo", "save_as_netcdf", "load_ssrl"]


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


def load_ssrl(filename):
    try:
        filename = find_first_file(filename)
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
        "CreationTime": "creation_time",
        "HDF5Version": "HDF5_Version",
        "H5pyVersion": "h5py_version",
        "Notes": "description",
        "Sample": "sample",
        "User": "user",
        "Model": "model",
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
    }
    coords_keys_mapping = {
        "Kinetic Energy": "eV",
        "ThetaX": "phi",
        "ThetaY": "theta",
    }

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
            data = data.assign_coords(
                {
                    ax["Label"]: np.linspace(ax["Minimum"], ax["Maximum"], ax["Count"])
                    for ax in axes
                }
            )
            data = data.rename_vars(Count="spectrum", Time="time")

    for k in list(attrs.keys()):
        if k in attr_keys_mapping.keys():
            attrs[attr_keys_mapping[k]] = attrs.pop(k)

    data = data.rename({k: v for k, v in coords_keys_mapping.items() if k in data.dims})

    if "theta" not in itertools.product(attrs.keys(), data.dims):
        attrs["theta"] = 0.0

    attrs["alpha"] = 90.0
    attrs["psi"] = 0.0

    for a in ["alpha", "beta", "theta", "chi", "phi", "psi"]:
        try:
            data = data.assign_coords({a: np.deg2rad(data[a])})
        except KeyError:
            data = data.assign_coords({a: np.deg2rad(attrs.pop(a))})
    data.attrs = attrs
    data.spectrum.attrs = attrs
    data.time.attrs = attrs
    return data


def find_first_file(file, allow_soft_match=False):
    workspace = arpes.config.CONFIG["WORKSPACE"]
    workspace_path = os.path.join(workspace["path"], "data")
    workspace = workspace["name"]
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
            files = endbase.files_for_search(dir)

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
