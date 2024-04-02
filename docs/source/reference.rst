*************
API Reference
*************

ERLabPy is organized into multiple subpackages and submodules.

Subpackages
===========

========================   ========================
Subpackage                 Description
========================   ========================
`erlab.analysis`           Data analysis
`erlab.io`                 Read & write ARPES data
`erlab.plotting`           Plot
`erlab.interactive`        Interactive plotting based on Qt and pyqtgraph
`erlab.characterization`   Analyze sample characterization results such as XRD and transport measurements
========================   ========================

.. currentmodule:: erlab

.. toctree::
   :hidden:

   erlab.analysis
   erlab.io
   erlab.plotting
   erlab.interactive
   erlab.characterization

Submodules
==========

==================  ==================
Submodule           Description
==================  ==================
`erlab.lattice`     Tools for working with real and reciprocal lattices.
`erlab.constants`   Physical constants and unit conversion
`erlab.accessors`   `xarray accessors <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_
`erlab.parallel`    Helpers for parallel processing
==================  ==================

.. toctree::
   :hidden:

   erlab.lattice
   erlab.constants
   erlab.accessors
   erlab.parallel
