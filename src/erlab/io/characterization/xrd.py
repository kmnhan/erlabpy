"""Functions related to loading x-ray diffraction spectra.

Currently only supports loading raw data from igor ``.itx`` files.

"""

__all__ = ["load_xrd_itx"]

import ast
import re

import numpy as np
import xarray as xr


def load_xrd_itx(path: str, **kwargs):
    r"""Load x-ray diffraction spectra from ``.itx`` file for Igor pro.

    Parameters
    ----------
    path
        Local path to ``.itx`` file.
    **kwargs
        Extra arguments to `open`.

    Returns
    -------
    xarray.Dataset
        Dataset object containing data from the file.

    Note
    ----
    By default, the file is read with the ``'windows-1252'`` encoding. This
    behavior can be customized by supplying keyword arguments.

    Examples
    --------
    Load from file:

    >>> xrd_data = load_xrd_itx("/path/to/example_data.itx")
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
    kwargs.setdefault("encoding", "windows-1252")
    with open(path, **kwargs) as file:
        content = file.read()

    search = re.search(r"IGOR\nWAVES/O\s(.*?)\nBEGIN\n(.+?)\nEND", content, re.DOTALL)
    if search is None:
        raise ValueError("Failed to parse .itx file.")

    head, data = search.groups()
    head = head.split(", ")

    data = np.array(
        ast.literal_eval("[[" + data.replace("\n", "],[").replace(" ", ",") + "]]")
    )
    ds = xr.Dataset({head[i]: ([head[0]], data[:, i]) for i in range(len(head))})
    if "diff" in ds.data_vars:
        ds = ds.rename_vars(diff="residual")
    return ds
