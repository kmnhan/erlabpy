"""Functions related to analyzing temperature-dependent resistance data.

Currently only supports loading raw data from `.dat` files output by
physics lab III equipment.

"""
import numpy as np
import xarray as xr

__all__ = ['load_resistance_physlab']

def load_resistance_physlab(path,encoding='windows-1252',**kwargs):
    """Loads resistance measurement acquired with physics lab III equipment.

    Parameters
    ----------
    path : str
        Local path to `.dat` file.
    encoding : str, optional
        Open file with given encoding, default `'windows-1252'` when optional.

    Returns
    -------
    out : xarray.Dataset object
        Dataset object containing resistance data from the file.
        
    """
    data = np.genfromtxt(
        path,
        delimiter = '\t',
        skip_header = 3,
        dtype = float,
        encoding = encoding,
        usecols = [2,3,4,5,6,7]
    )
    head = ['time','temp','res','curr','temperr','reserr']
    ds = xr.Dataset({head[i]:([head[1]],data[:,i]) for i in range(len(head))})
    return ds