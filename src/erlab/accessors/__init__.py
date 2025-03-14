"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

``erlab`` provides a collection of accessors for convenient data analysis and
visualization. The source code is organized into several modules:

.. currentmodule:: erlab.accessors

.. autosummary::
   :toctree: generated

   utils
   general
   kspace
   fit

However, users should not import these modules directly. Instead, the accessors are
registered with :mod:`xarray` and can be accessed like attributes of
:class:`xarray.Dataset` and :class:`xarray.DataArray` objects.

All available accessor methods and attributes defined are documented below.

"""  # noqa: D205
