"""
Read & write ARPES data.

.. currentmodule:: erlab.io

Storing and retrieving files using `pickle` is quick and convenient, but pickled files
can only be opened in the environment it was saved in. This module provides alternatives
that enables saving and loading data in an efficient way, without compromising
efficiency.

This module also provides functions that enables loading various files such as igor pro
files, livexy and livepolar files from Beamline 4.0.1 of the Advanced Light Source, and
raw data from the Stanford Synchrotron Light Source Beamline 5-2.

Modules
=======

.. autosummary::
   :toctree: generated
   
   utilities
   igor
   merlin
   ssrl52

"""

__all__ = [
    "load_experiment",
    "load_wave",
    "da30",
    "load_als_bl4",
    "load_ssrl",
    "load_live",
    "open_hdf5",
    "load_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
]

import warnings

from erlab.io.utilities import load_hdf5, open_hdf5, save_as_hdf5, save_as_netcdf
from erlab.io.igor import load_experiment, load_wave

from erlab.io import merlin, ssrl52, da30

load_als_bl4 = merlin.load
load_ssrl = ssrl52.load


def load_igor_ibw(*args, **kwargs):
    warnings.warn("Use `erlab.io.load_wave` instead", DeprecationWarning, stacklevel=2)
    return load_wave(*args, **kwargs)


def load_igor_pxp(*args, **kwargs):
    warnings.warn(
        "Use `erlab.io.load_experiment` instead", DeprecationWarning, stacklevel=2
    )
    return load_experiment(*args, **kwargs)
