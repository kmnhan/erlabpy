**********
User Guide
**********

This section contains some examples for getting started with ARPES data analysis and visualization. For the full list of packages and modules provided by ERLabPy, see :doc:`reference`.


Introduction
============

The following documentation assumes basic python programming experience. If you are not familiar with manipulating numpy arrays, `the beginner's guide to numpy <https://numpy.org/doc/stable/user/absolute_beginners.html>`_ is a great place to start. Data in ERLabPy are mostly represented by :mod:`xarray` objects. The user guide of the `xarray documentation <https://docs.xarray.dev/en/stable/index.html>`_ provides an excellent overview on data manipulation.

.. toctree::
   notebooks/io
   notebooks/indexing
   notebooks/plotting

Originally a collection of personal plotting macros as a complement to PyARPES, ERLabPy has evolved to contain its own data loading functions and more. Although PyARPES is a powerful platform, it has its limitations, such as the heavy provenance and logging overhead and angle coordinate conventions. Unlike PyARPES which aims to provide entire workflows for ARPES data analysis, ERLabPy tries to provide the user with convenient methods that any python programmer can easily integrate into their routine, many of which are not limited to the field of ARPES. In the near future, ERLabPy aims to be a self-consistent package that aids ARPES data analysis, but currently it still depends on many PyARPES features. The following are a list of features that still depend on PyARPES.

WIP