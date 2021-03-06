"""Functions related to analyzing temperature-dependent resistance data.

Currently only supports loading raw data from `.dat` files output by
physics lab III equipment.

"""
import re
from io import StringIO

import numpy as np
import xarray as xr

__all__ = ['load_resistance_physlab']

def load_resistance_physlab(path:str, encoding='windows-1252', **kwargs):
    """Loads resistance measurement acquired with physics lab III
    equipment.

    Parameters
    ----------
    path : str
        Local path to `.dat` file.
    encoding : str, optional
        Open file with given encoding, default `'windows-1252'` when
        optional.

    Returns
    -------
    ds : xarray.Dataset object
        Dataset object containing resistance data from the file.

    Examples
    --------
    Load from file:

    >>> data = load_resistance_physlab('/path/to/example_data.dat')
    >>> data
    <xarray.Dataset>
    Dimensions:  (temp: 1087)
    Coordinates:
    * temp     (temp) float64 21.44 21.61 21.81 ... 299.4 299.6 299.8
    Data variables:
        time     (temp) float64 44.45 66.06 ... 2.346e+04 2.348e+04
        res      (temp) float64 0.02837 0.02886 ... 0.05812 0.05914
        curr     (temp) float64 0.0001 0.0001 ... 0.0001 0.0001
        temperr  (temp) float64 0.098 0.11 ... 0.066 0.046
        reserr   (temp) float64 0.001551 0.001202 ... 0.001027 0.001125
    
    Plot resistance versus temperature:

    >>> data.res.plot()

    """
    with open(path, 'r', encoding=encoding) as file:
        content = re.sub(r"(e[-+]\d{3}) {2,3}(-?)",
                     "\\g<1>\\t \\g<2>", file.read(), 0, re.MULTILINE)
    data = np.genfromtxt(
        StringIO(content),
        delimiter="\t",
        skip_header=3,
        usecols=[2,3,4,5,6,7],
        **kwargs
    )
    head = ['time','temp','res','curr','temperr','reserr']
    ds = xr.Dataset({head[i]:([head[1]],data[:,i]) for i in range(len(head))})
    return ds