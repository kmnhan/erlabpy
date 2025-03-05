"""Read & write ARPES data.

.. currentmodule:: erlab.io

This module provides functions that enables loading various files such as hdf5 files,
igor pro files, and ARPES data from different beamlines and laboratories.

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


.. rubric:: Modules

.. autosummary::
   :toctree: generated

   plugins
   dataloader
   utils
   igor
   nexusutils
   exampledata
   characterization

.. rubric:: Module Attributes

.. attribute:: loaders

   A global registry of all loaders registered in the session. The keys are the names of
   the loaders and the values are the loader objects.

   .. seealso::

      :func:`set_loader`, :func:`set_data_dir`, :func:`loader_context`

.. rubric:: Functions

.. autosummary::

   load
   loader_context
   set_data_dir
   set_loader
   extend_loader
   summarize

"""

import warnings

import lazy_loader as _lazy

import erlab.io.plugins  # noqa: F401

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


def load_wave(*args, **kwargs):
    from erlab.io.igor import load_wave as _load_wave

    warnings.warn("Use `xarray.open_dataarray` instead", FutureWarning, stacklevel=2)
    return _load_wave(*args, **kwargs)


def load_experiment(*args, **kwargs):
    from erlab.io.igor import load_experiment as _load_experiment

    warnings.warn("Use `xarray.open_dataset` instead", FutureWarning, stacklevel=2)
    return _load_experiment(*args, **kwargs)
