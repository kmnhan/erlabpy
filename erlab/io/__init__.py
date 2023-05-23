"""Functions for data IO.

Storing and retrieving files using `pickle` is quick and convenient, but pickled files
can only be opened in the environment it was saved in. This module provides alternatives
that enables saving and loading data in an efficient way, without compromising
efficiency.

This module also provides functions that enables loading various files such as igor pro
files, livexy and livepolar files from Beamline 4.0.1 of the Advanced Light Source, and
raw data from the Stanford Synchrotron Light Source Beamline 5-2.

"""
import warnings

from erlab.io.igor import load_experiment, load_wave, load_pxp, load_ibw
from erlab.io.merlin import load as load_als_bl4
from erlab.io.merlin import load_livepolar, load_livexy
from erlab.io.ssrl52 import load as load_ssrl
from erlab.io.utilities import load_hdf5, open_hdf5, save_as_hdf5, save_as_netcdf

__all__ = [
    "load_pxp",
    "load_ibw",
    "load_experiment",
    "load_wave",
    "load_als_bl4",
    "load_livepolar",
    "load_livexy",
    "open_hdf5",
    "load_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
    "load_ssrl",
]


def load_igor_ibw(*args, **kwargs):
    warnings.warn("Use `erlab.io.load_wave` instead", DeprecationWarning, stacklevel=2)
    return load_wave(*args, **kwargs)


def load_igor_pxp(*args, **kwargs):
    warnings.warn(
        "Use `erlab.io.load_experiment` instead", DeprecationWarning, stacklevel=2
    )
    return load_experiment(*args, **kwargs)
