# API Reference

ERLabPy is organized into multiple subpackages and submodules classified by their functionality. The following table lists the subpackages and submodules of ERLabPy.

## Subpackages

| Subpackage               | Description                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| {mod}`erlab.analysis`    | Routines for analyzing ARPES data.                                                                                                       |
| {mod}`erlab.io`          | Reading and writing data.                                                                                                                |
| {mod}`erlab.plotting`    | Functions related to static plotting with matplotlib.                                                                                    |
| {mod}`erlab.interactive` | Interactive tools and widgets based on Qt and pyqtgraph                                                                                  |
| {mod}`erlab.accessors`   | [xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html). You will not need to import this module directly. |
| {mod}`erlab.utils`       | Utility functions and classes, typically used internally.                                                                                |

```{eval-rst}
.. currentmodule:: erlab
```

```{toctree}
:hidden: true

erlab.analysis
erlab.io
erlab.plotting
erlab.interactive
erlab.accessors
erlab.utils
```

## Submodules

| Submodule              | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| {mod}`erlab.lattice`   | Tools for working with real and reciprocal lattices.  |
| {mod}`erlab.constants` | Physical constants and functions for unit conversion. |

```{toctree}
:hidden: true

erlab.lattice
erlab.constants
```
