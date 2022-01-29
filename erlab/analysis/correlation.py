import numpy as np
import xarray as xr
from scipy.fft import fftshift, fft2, ifft2
from scipy.signal import correlate, correlation_lags
from joblib import delayed, Parallel

__all__ = ['acf2','match_dims','xcorr1']

def acf2rect(IMG:np.ndarray):
    """
    Autocorrelation of rectangular image.
    """
    [M,N] = np.shape(IMG)
    IMG_p = np.zeros(shape=(2*M-1,2*N-1))
    IMG_p[:M,:N] = IMG
    ACF = np.abs(fftshift(ifft2(np.abs(fft2(IMG_p))**2)))
    with np.errstate(divide='ignore',invalid='ignore'):
        ACF = ACF / ACF[M,N]
    return ACF

def acf2_arr(IMG:np.ndarray):
    """
    Autocorrelation of masked image.
    """
    ACF = acf2rect(~np.isnan(IMG))
    ACF[ACF<=1e-10] = np.nan
    with np.errstate(invalid='ignore'):
        ACF = acf2rect(np.nan_to_num(IMG)) / ACF
    return ACF

def xacf2(da:xr.DataArray,parallel=True,n_jobs=-1,verbose=0) -> xr.DataArray:
    """
    Slicewise autocorrelation function of constant energy maps in volumetric data.
    """
    # TODO: incorporate backend scipy.signal.correlate
    if isinstance(da, xr.Dataset):
        da = da.spectrum
    ndims = len(da.dims)
    if ndims == 3:
        da = da.transpose('ky','eV','kx')
        ny, nE, nx = da.shape
    elif ndims == 2:
        da = da.transpose('ky','kx')
        ny, nx = da.shape
    else:
        raise ValueError('The input to acf2 must be 2D or 3D.')
    vol = da.values
    dx, dy = (da.kx[1]-da.kx[0]).values, (da.ky[1]-da.ky[0]).values
    qx = np.linspace(0, 2*(nx-1)*dx, 2 * nx- 1) - (nx-1)*dx
    qy = np.linspace(0, 2*(ny-1)*dy, 2 * ny- 1) - (ny-1)*dy
    if ndims == 3:
        if parallel is True:
            sub_arrays = Parallel(n_jobs=n_jobs,prefer="threads",verbose=verbose)(
                                delayed(acf2_arr)(vol[:,i,:])
                                for i in range(vol.shape[1]))
            acf = np.stack(sub_arrays, axis=1)
        else:
            acf = np.full(
                shape=(2 * ny - 1, nE, 2 * nx - 1),
                fill_value=np.nan,
                dtype=np.float64
            )
            for i in range(acf.shape[1]):
                acf[:,i,:] = acf2_arr(vol[:,i,:])
        acfarray = xr.DataArray(acf,dims=['qy','eV','qx'],coords={'qy':qy,'eV':da.eV,'qx':qx})
        try:
            acfarray.eV.attrs['units'] = da.eV.units
        except AttributeError:
            pass
    else:
        acf = acf2_arr(vol)
        acfarray = xr.DataArray(acf,dims=['qy','qx'],coords={'qy':qy,'qx':qx})
    try:
        acfarray.qx.attrs['units'] = da.kx.units
        acfarray.qy.attrs['units'] = da.ky.units
    except AttributeError:
        pass
    return acfarray

def match_dims(da1:xr.DataArray,da2:xr.DataArray):
    """
    Returns the second array interpolated with the coordinates of the first array, making them the same size.
    """
    return da2.interp({dim:da1[dim] for dim in da2.dims})

def xcorr1d(in1:xr.DataArray,in2:xr.DataArray):
    """
    Performs 1-dimensional correlation analysis on `xarray.DataArray`s.
    """
    in2 = match_dims(in1,in2)
    out = in1.copy(deep=True)
    xind = correlation_lags(in1.values.size,in2.values.size,mode='same')
    xzero = np.flatnonzero(xind == 0)[0]
    out.values = correlate(in1.fillna(0).values,in2.fillna(0).values,mode='same',method='direct')
    out[in1.dims[0]] = out[in1.dims[0]] - out[in1.dims[0]][xzero]
    return out