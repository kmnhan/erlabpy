import numpy as np
import xarray as xr
from astropy.io import fits

__all__ = ['showfitsinfo','save_as_netcdf']

def showfitsinfo(path:str):
    """
    Prints raw metadata from FITS file.
    """
    with fits.open(path, ignore_missing_end=True) as hdul:
        hdul.verify("silentfix+warn")
        hdul.info()
        for i in range(len(hdul)):
            # print(f'\nColumns in {i:d}: {hdul[i].columns.names!r}')
            print(f'\nHeaders in {i:d}:\n{hdul[i].header!r}')
            
def fix_attr_format(da:xr.DataArray):
    """
    Discards attributes that are incompatible with the NetCDF5 file format.
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

def save_as_netcdf(data:xr.DataArray,filename:str):
    data = data.assign_attrs(provenance='')
    fix_attr_format(data).to_netcdf(filename, encoding={var: dict(zlib=True, complevel=5) for var in data.coords})

def save_as_fits():
    # TODO
    pass