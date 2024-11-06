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
   utils
   igor
   nexusutils
   exampledata
   characterization


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

- Set the default loader for the session:

  >>> erlab.io.set_loader("merlin")

Learn more about loaders in the :ref:`User Guide <loading-arpes-data>`.

"""

__all__ = [
    "load",
    "load_hdf5",
    "loader_context",
    "loaders",
    "open_hdf5",
    "save_as_hdf5",
    "save_as_netcdf",
    "set_data_dir",
    "set_loader",
    "summarize",
]

import warnings

from erlab.io.dataloader import LoaderRegistry
from erlab.io.utils import load_hdf5, open_hdf5, save_as_hdf5, save_as_netcdf

# Import plugins last
# isort: off
import erlab.io.plugins  # noqa: F401

loaders = LoaderRegistry.instance()
"""
Global instance of :class:`LoaderRegistry <erlab.io.dataloader.LoaderRegistry>`.

:meta hide-value:
"""

load = loaders.load
loader_context = loaders.loader_context
set_data_dir = loaders.set_data_dir
set_loader = loaders.set_loader
summarize = loaders.summarize


def load_wave(*args, **kwargs):
    from erlab.io.igor import load_wave as _load_wave

    warnings.warn(
        "Use `xarray.open_dataarray` instead", DeprecationWarning, stacklevel=2
    )
    return _load_wave(*args, **kwargs)


def load_experiment(*args, **kwargs):
    from erlab.io.igor import load_experiment as _load_experiment

    warnings.warn("Use `xarray.open_dataset` instead", DeprecationWarning, stacklevel=2)
    return _load_experiment(*args, **kwargs)
