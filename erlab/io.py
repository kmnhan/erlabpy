"""Functions for data IO.

PyARPES stores files using `pickle`, which can only be opened in the
environment it was saved in.
The function `save_as_netcdf` saves an array in the `netCDF4` format,
which can be opened in Igor as `hdf5` files. 

"""

import numpy as np
import xarray as xr
from astropy.io import fits

__all__ = ['showfitsinfo', 'save_as_netcdf']

def showfitsinfo(path:str):
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
            print(f'\nHeaders in {i:d}:\n{hdul[i].header!r}')

def fix_attr_format(da:xr.DataArray):
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
    valid_dtypes = ['S1', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f4', 'f8']
    for key in da.attrs.keys():
        isValid = 0
        for dt in valid_dtypes:
            isValid+=(np.array(da.attrs[key]).dtype == np.dtype(dt))
        if not isValid:
            try:
                da = da.assign_attrs({key:str(da.attrs[key])})
            except:
                da = da.assign_attrs({key:''})
    return da

def save_as_netcdf(data:xr.DataArray, filename:str, **kwargs):
    """Saves data in 'netCDF4' format.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray object to save.
    **kwargs : dict, optional
        Extra arguments to `DataArray.to_netcdf`: refer to the xarray
        documentation for a list of all possible arguments.

    """
    data = data.assign_attrs(provenance='')
    fix_attr_format(data).to_netcdf(
        filename,
        encoding={var: dict(zlib=True, complevel=5) for var in data.coords},
        **kwargs
    )

def save_as_fits():
    # TODO
    pass