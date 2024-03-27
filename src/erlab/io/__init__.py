"""Read & write ARPES data.

.. currentmodule:: erlab.io

This module provides functions that enables loading various files such as hdf5 files,
igor pro files, and ARPES data from different beamlines and laboratories.

Modules
=======

.. autosummary::
   :toctree: generated

   plugins
   dataloader
   utilities
   igor


For a single session, it is very common to use only one type of loader for a single
folder with all your data. Hence, the module provides a way to set a default loader for
a session. This is done using the :func:`set_loader` function. The same can be done for
the data directory using the :func:`set_data_dir` function.

For instructions on how to write a custom loader, see
:doc:`/generated/erlab.io.dataloader`.


Examples
--------

- View all registered loaders:

  >>> erlab.io.loaders

- Load data by explicitly specifying the loader:

  >>> dat = erlab.io.loaders["merlin"].load(...)


"""

__all__ = [
    "loaders",
    "set_loader",
    "loader_context",
    "set_data_dir",
    "load",
    "summarize",
    "load_experiment",
    "load_wave",
    "load_live",
    "open_hdf5",
    "load_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
]

import warnings

from erlab.io.dataloader import (
    load,
    loader_context,
    loaders,
    set_data_dir,
    set_loader,
    summarize,
)
from erlab.io.igor import load_experiment, load_wave
from erlab.io.utilities import load_hdf5, open_hdf5, save_as_hdf5, save_as_netcdf

# Import plugins last
# isort: off
import erlab.io.plugins  # noqa: F401

merlin = loaders["merlin"]
ssrl52 = loaders["ssrl52"]


def load_igor_ibw(*args, **kwargs):
    warnings.warn("Use `erlab.io.load_wave` instead", DeprecationWarning, stacklevel=2)
    return load_wave(*args, **kwargs)


def load_igor_pxp(*args, **kwargs):
    warnings.warn(
        "Use `erlab.io.load_experiment` instead", DeprecationWarning, stacklevel=2
    )
    return load_experiment(*args, **kwargs)
