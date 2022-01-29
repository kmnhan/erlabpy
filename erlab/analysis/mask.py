import numpy as np 
import xarray as xr
from arpes.analysis.mask import polys_to_mask

__all__ = ['mask_with_hex_bz']

def mask_with_hex_bz(kxymap:xr.DataArray, a=3.54, rotate=0,radius=None,invert=True):
    """
    Returns array masked with a hexagonal BZ.
    """
    if isinstance(kxymap, xr.Dataset):
        kxymap = kxymap.spectrum

    dims = kxymap.dims
    assert len(dims) in (2,3), 'Input must be 2D or 3D'
    rotate = np.radians(rotate)
    

    if 'kx' in dims:
        dimnames = ('ky','kx')
    elif 'qx' in dims:
        dimnames = ('qy','qx')

    l = 2 * np.pi / (a * 3)
    angles = np.radians([0,60,120,180,240,300])
    mask = {
        'dims': dimnames,
        'polys': [[[2*l*np.sin(rotate+theta), 2*l*np.cos(rotate+theta)] for theta in angles]]
    }
    if len(dims) == 3:
        kxycut = kxymap.isel(eV=0)
    else: 
        kxycut = kxymap
    mask_arr = polys_to_mask(
        mask,
        kxycut.coords,
        [s for i, s in enumerate(kxycut.shape) if kxycut.dims[i] in mask['dims']],
        radius = radius,
        invert = invert
    )
    if len(dims) == 3:
        mask_arr = np.repeat(mask_arr[:,np.newaxis,:], kxymap.eV.size, axis=1)
    masked = kxymap.copy(deep=True)
    masked.values = masked.values * 1.0
    masked.values[mask_arr] = np.nan
    return masked