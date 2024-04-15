**********
User Guide
**********

.. admonition:: Work in Progress
   :class: warning

   The user guide is incomplete. For the full list of packages and modules provided by
   ERLabPy, see :doc:`../reference`.

Introduction
============

Welcome to the ERLabPy user guide! This guide provides an overview of ERLabPy and its
core features.

If you are new to programming with python, `Scientific Python Lectures
<https://github.com/jrjohansson/scientific-python-lectures>`_ is a great place to start.

Data structures in ERLabPy are represented using `xarray <https://docs.xarray.dev/>`_\
:cite:p:`hoyer2017xarray`, which provides a powerful data structure for working with
multi-dimensional arrays. Visit the `xarray tutorial <https://tutorial.xarray.dev/>`_
and the `xarray user guide <https://docs.xarray.dev/en/stable/index.html>`_ to get
familiar with xarray.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   io
   indexing
   plotting
   kconv
   curve-fitting
   imagetool

Further reading
===============

- `Lectures on scientific computing with Python
  <https://github.com/jrjohansson/scientific-python-lectures>`_
- `The beginner's guide to numpy
  <https://numpy.org/doc/stable/user/absolute_beginners.html>`_
- `Xarray tutorial <https://tutorial.xarray.dev/>`_
