"""Functions related to loading temperature-dependent resistance data.

Currently only supports loading raw data from ``.dat`` and ``.csv`` files output by
physics lab III equipment.

"""

import os
import re
from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["load_resistance_physlab"]


def load_resistance_physlab(
    path: str,
    get: Literal["full", "cool", "heat"] = "full",
    as_temperature: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Load resistance measurement acquired with physics lab III equipment.

    Parameters
    ----------
    path
        Local path to ``.dat`` or ``.csv`` file.
    get
        For data acquired after July 2024, you can choose to only load the data acquired
        during heating or cooling. Defaults to the full dataset.
    as_temperature
        If ``True``, the dataset is returned with temperature as the index, and all
        values are sorted by temperature. If ``False``, the dataset is returned in a
        form that retains the original order of the data. Defaults to ``False``.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing resistance data from the file.

    """
    if os.path.splitext(path)[1] == ".dat":
        out = _load_resistance_physlab_old(path, **kwargs)
    else:
        out = (
            pd.read_csv(path, index_col=0, usecols=[1, 2, 3, 4])
            .to_xarray()
            .rename(
                {
                    "Elapsed Time (s)": "time",
                    "Temperature (K)": "temp",
                    "Resistance (Ohm)": "res",
                    "Current (A)": "curr",
                }
            )
        )

        if get != "full":
            delim_time = out.time.where(out.res.isnull(), drop=True).values
            if get == "cool":
                out_sel = out.where(out.time < delim_time, drop=True)
            else:
                out_sel = out.where(out.time > delim_time, drop=True)
            if out_sel.time.size != 0:
                out = out_sel

    if as_temperature:
        return out.swap_dims({"time": "temp"}).sortby("temp")
    return out


def _load_resistance_physlab_old(
    path: str, encoding: str = "windows-1252", **kwargs
) -> xr.Dataset:
    """Load resistance measurement acquired with physics lab III equipment.

    Parameters
    ----------
    path
        Local path to ``.dat`` file.
    encoding
        Encoding passed onto `open`, defaults to ``'windows-1252'``.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing resistance data from the file.

    Examples
    --------
    Load from file:

    >>> data = load_resistance_physlab("/path/to/example_data.dat")
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
    content = re.sub(
        r"(e[-+]\d{3}) {2,3}(-?)",
        "\\g<1>\\t \\g<2>",
        Path(path).read_text(encoding),
        count=0,
        flags=re.MULTILINE,
    )

    content = content.replace("-1.#IO", "   nan")
    data = np.genfromtxt(
        StringIO(content),
        delimiter="\t",
        skip_header=3,
        usecols=[2, 3, 4, 5, 6, 7],
        **kwargs,
    )
    return xr.Dataset(
        data_vars={
            "temp": ("time", data[:, 1]),
            "res": ("time", data[:, 2]),
            "curr": ("time", data[:, 3]),
            "temp_err": ("time", data[:, 4]),
            "res_err": ("time", data[:, 5]),
        },
        coords={"time": data[:, 0]},
    )
