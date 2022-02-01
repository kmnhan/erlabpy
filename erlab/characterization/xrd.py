"""Functions related to analyzing x ray diffraction spectra. 

Currently only supports loading raw data from igor `.itx` files, but
this is enough for only plotting with `matplotlib`.

"""
import ast, re

import numpy as np
import xarray as xr

__all__ = ['load_xrd_itx']

def load_xrd_itx(path, **kwargs):
    """Load x-ray diffraction spectra from .itx file for Igor pro. 

    Parameters
    ----------
    path : str
        Local path to `.itx` file.
    **kwargs : dict, optional
        Extra arguments to `open`: refer to the official Python
        documentation for a list of all possible arguments.
 
    Returns
    -------
    out : xarray.Dataset object
        Dataset object containing data from the file.
        
    """

    with open(path,'r',**kwargs) as file:
        content = file.read()
    head, data = re.search(r'IGOR\nWAVES/O\s(.*?)\nBEGIN\n(.+?)\nEND',
                           content, re.DOTALL).groups()
    head = head.split(', ')
    data = np.array(
        ast.literal_eval(
            '[['+ data.replace('\n','],[').replace(' ',',')+']]'
        )
    )
    ds = xr.Dataset(
            {head[i]:([head[0]],data[:,i])for i in range(len(head))}
    )
    return ds