"""Functions related to analyzing x ray diffraction spectra. 

Currently only supports loading raw data from igor `.itx` files, but
this is enough for only plotting with `matplotlib`.

"""
import ast
import re

import numpy as np
import xarray as xr

__all__ = ['load_xrd_itx']

def load_xrd_itx(path:str, **kwargs):
    r"""Load x-ray diffraction spectra from .itx file for Igor pro. 

    Parameters
    ----------
    path : str
        Local path to `.itx` file.
    **kwargs : dict, optional
        Extra arguments to `open`: refer to the official Python
        documentation for a list of all possible arguments.
 
    Returns
    -------
    ds : xarray.Dataset object
        Dataset object containing data from the file.
    
    Notes
    -----
    By default, the file is read with the `windows-1252` encoding. This
    behavior can be customized with `**kwargs`.

    Examples
    --------
    Load from file:

    >>> xrd_data = load_xrd_itx('/path/to/example_data.itx')
    >>> xrd_data
    <xarray.Dataset>
    Dimensions:   (twotheta: 6701)
    Coordinates:
    * twotheta  (twotheta) float64 3.0 3.01 3.02 ... 69.98 69.99 70.0
    Data variables:
        yobs      (twotheta) float64 143.0 163.0 ... 7.0 7.0 7.0 2.0
        ycal      (twotheta) float64 119.4 118.8 ... 5.316 5.351 5.387
        bkg       (twotheta) float64 95.31 94.89 ... 5.228 5.264 5.3
        diff      (twotheta) float64 23.61 44.19 ... 1.684 1.649 -3.387
    
    Plot observed data:

    >>> xrd_data.yobs.plot()

    """
    kwargs.setdefault('encoding','windows-1252')
    with open(path, 'r', **kwargs) as file:
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
