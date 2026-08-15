# Reference

Use this section while you work. It describes the Python API, GUI applications,
dependencies, and product interfaces.

```{toctree}
:caption: Reference
:maxdepth: 2

reference/installation
reference/python-and-gui-map
reference/gui/index
```

## Python API

The Python API reference describes ERLabPy subpackages and modules by function.

The {mod}`erlab.analysis.fit.models` API is the catalog of predefined fitting models.
The {doc}`curve-fitting How-to guides <how-to/python/curve-fitting>` contain procedures
for fitting measured spectra and inspecting the results.

### Data input and loaders

Use the {mod}`erlab.io` Reference to look up the loader registry and common loading
functions. The following module pages describe the available machinery:

- {mod}`erlab.io.plugins` lists the built-in endstation loaders.
- {mod}`erlab.io.igor` describes the Igor Pro xarray backend and export functions.
- {mod}`erlab.io.metadata` describes spreadsheet-backed metadata sources.
- {mod}`erlab.io.fitsutils` describes FITS helpers.
- {mod}`erlab.io.nexusutils` describes NeXus helpers.

For actions on experimental files, use the {doc}`data loading and saving How-to guides
<how-to/python/loading-and-saving>`.

### Other file formats

Use the file-format library first when no endstation loader applies:

- Use {func}`xarray.open_dataarray`, {func}`xarray.open_dataset`, or
  {func}`xarray.open_datatree` for NetCDF and supported HDF5 files.
- Use {func}`xarray.open_groups` to inspect an HDF5 file that contains an unknown group
  structure.
- Use {func}`pandas.read_csv` or {func}`pandas.read_excel` for tabular data. Convert the
  indexed table with {meth}`pandas.DataFrame.to_xarray`.
- Use {func}`erlab.io.fitsutils.fits_to_xarray` for supported FITS data.
- Use {mod}`erlab.io.nexusutils` for NeXus structures that need explicit conversion.

The result must still follow the {ref}`ARPES data conventions <data-conventions>` before
you use an ARPES-specific analysis routine. A new endstation format usually needs a
{doc}`loader plugin <contributing/loaders>` so that loading and metadata normalization
remain reproducible.

## Subpackages

| Subpackage               | Description                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| {mod}`erlab.analysis`    | Routines for analyzing ARPES data.                                                                                                       |
| {mod}`erlab.extensions`  | Public APIs for ImageTool Manager analysis and loader extensions.                                                                        |
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
erlab.extensions
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
