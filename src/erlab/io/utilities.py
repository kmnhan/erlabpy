import os
import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

__all__ = [
    "showfitsinfo",
    "get_files",
    "open_hdf5",
    "load_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
]


def showfitsinfo(path: str | os.PathLike):
    """Prints raw metadata from a ``.fits`` file.

    Parameters
    ----------
    path
        Local path to ``.fits`` file.

    """
    from astropy.io import fits

    with fits.open(path, ignore_missing_end=True) as hdul:
        hdul.verify("silentfix+warn")
        hdul.info()
        for i in range(len(hdul)):
            # print(f'\nColumns in {i:d}: {hdul[i].columns.names!r}')
            print(f"\nHeaders in {i:d}:\n{hdul[i].header!r}")


def get_files(
    directory,
    extensions: Sequence[str] | None = None,
    contains: str | None = None,
    notcontains: str | None = None,
) -> list[str]:
    """Returns a list of files in a directory with the given extensions.

    Parameters
    ----------
    directory
        Target directory.
    extensions
        List of extensions to filter for. If not provided, all files are returned.
    contains
        String to filter for in the file names.
    notcontains
        String to filter out of the file names.

    Returns
    -------
    files : list of str
        List of files in the directory.

    """

    files = []

    for f in os.listdir(directory):
        if extensions is not None and os.path.splitext(f)[1] not in extensions:
            continue
        if contains is not None and contains not in f:
            continue
        if notcontains is not None and notcontains in f:
            continue
        files.append(os.path.join(directory, f))

    return files


def fix_attr_format(da: xr.DataArray):
    """Discards attributes that are incompatible with the ``netCDF4`` file format.

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
    for key in da.attrs:
        isValid = 0
        for dt in valid_dtypes:
            isValid += np.array(da.attrs[key]).dtype == np.dtype(dt)
        if not isValid:
            try:
                da = da.assign_attrs({key: str(da.attrs[key])})
                warnings.warn(
                    f"The attribute {key} with invalid type {dt}"
                    " will be converted to string",
                    stacklevel=1,
                )
            except TypeError:
                # this is VERY unprobable...
                da = da.assign_attrs({key: ""})
                warnings.warn(
                    f"The attribute {key} with invalid type {dt} will be removed",
                    stacklevel=1,
                )
    return da


def open_hdf5(filename: str | os.PathLike) -> xr.DataArray | xr.Dataset:
    try:
        return xr.open_dataarray(filename, engine="h5netcdf")
    except ValueError:
        return xr.open_dataset(filename, engine="h5netcdf")


def load_hdf5(filename: str | os.PathLike) -> xr.DataArray | xr.Dataset:
    try:
        return xr.load_dataarray(filename, engine="h5netcdf")
    except ValueError:
        return xr.load_dataset(filename, engine="h5netcdf")


def save_as_hdf5(
    data: xr.DataArray | xr.Dataset,
    filename: str | os.PathLike,
    igor_compat: bool = True,
    **kwargs: dict,
):
    """Saves data in ``HDF5`` format.

    Parameters
    ----------
    data
        `xarray.DataArray` to save.
    filename
        Target file name.
    igor_compat
        Make the resulting file compatible with Igor's `HDF5OpenFile`.
    **kwargs
        Extra arguments to `xarray.DataArray.to_netcdf`: refer to the `xarray`
        documentation for a list of all possible arguments.

    """
    kwargs.setdefault("engine", "h5netcdf")
    kwargs.setdefault("invalid_netcdf", True)

    data = data.copy(deep=True)
    for k, v in data.attrs.items():
        if v is None:
            data = data.assign_attrs({k: "None"})
        if isinstance(v, dict):
            data = data.assign_attrs({k: str(v)})
    if isinstance(data, xr.Dataset):
        igor_compat = False
    if igor_compat:
        # IGORWaveScaling order: chunk row column layer
        scaling = [[1, 0]]
        for i in range(data.ndim):
            coord = data[data.dims[i]].values
            delta = coord[1] - coord[0]
            scaling.append([delta, coord[0]])
        if data.ndim == 4:
            scaling[0] = scaling.pop(-1)
        data.attrs["IGORWaveScaling"] = scaling
    data.to_netcdf(
        filename,
        encoding={
            var: {"compression": "gzip", "compression_opts": 9} for var in data.coords
        },
        **kwargs,
    )


def save_as_netcdf(data: xr.DataArray, filename: str | os.PathLike, **kwargs: dict):
    """Saves data in ``netCDF4`` format.

    Discards invalid ``netCDF4`` attributes and produces a warning.

    Parameters
    ----------
    data
        `xarray.DataArray` to save.
    filename
        Target file name.
    **kwargs
        Extra arguments to `xarray.DataArray.to_netcdf`: refer to the `xarray`
        documentation for a list of all possible arguments.

    """
    # data = data.assign_attrs(provenance="")
    kwargs.setdefault("engine", "h5netcdf")
    fix_attr_format(data).to_netcdf(
        filename,
        encoding={var: {"zlib": True, "complevel": 5} for var in data.coords},
        **kwargs,
    )


def save_as_fits():
    # TODO
    raise NotImplementedError
