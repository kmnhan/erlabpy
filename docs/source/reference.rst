*************
API Reference
*************

ERLabPy is organized into multiple subpackages and submodules classified by their functionality. The following table lists the subpackages and submodules of ERLabPy.

Subpackages
===========

========================   ========================
Subpackage                 Description
========================   ========================
`erlab.analysis`           Routines for analyzing ARPES data.
`erlab.io`                 Reading and writing data.
`erlab.plotting`           Functions related to static plotting with matplotlib.
`erlab.interactive`        Interactive tools and widgets based on Qt and pyqtgraph
`erlab.accessors`          `xarray accessors <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_. You will not need to import this module directly.
`erlab.utils`              Utility functions and classes, typically used internally.
========================   ========================

.. currentmodule:: erlab

.. toctree::
   :hidden:

   erlab.analysis
   erlab.io
   erlab.plotting
   erlab.interactive
   erlab.accessors
   erlab.utils

Submodules
==========

==================  ==================
Submodule           Description
==================  ==================
`erlab.lattice`     Tools for working with real and reciprocal lattices.
`erlab.constants`   Physical constants and functions for unit conversion.
==================  ==================

.. toctree::
   :hidden:

   erlab.lattice
   erlab.constants
