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
{doc}`how-to/python/curve-fitting` contains procedures for fitting measured spectra and
inspecting the results.

### Data input and loaders

Use the {mod}`erlab.io` Reference to look up the loader registry and common loading
functions. The following module pages describe the available machinery:

- {mod}`erlab.io.plugins` lists the built-in endstation loaders.
- {mod}`erlab.io.igor` describes the Igor Pro xarray backend and export functions.
- {mod}`erlab.io.metadata` describes spreadsheet-backed metadata sources.
- {mod}`erlab.io.fitsutils` describes FITS helpers.
- {mod}`erlab.io.nexusutils` describes NeXus helpers.

For actions on experimental files, use {doc}`how-to/python/loading-and-saving`.

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
procedure from {doc}`contributing/loaders` so that loading and metadata normalization
remain reproducible.

## Plotting

(reference-plotting-styles)=

### Bundled style sheets

ERLabPy supplies the following Matplotlib style sheets. Styles that select a font
require that font on the system.

| Style sheet | Description |
| --- | --- |
| `erlab.general` | General-purpose ERLab figure defaults. |
| `erlab.nature` | Thin lines and tick sizes that resemble Springer Nature journal figures. |
| `erlab.arial` | Arial text and math fonts. |
| `erlab.times` | Times New Roman text and the STIX math font. |
| `erlab.helvetica` | Helvetica text and math fonts. |
| `erlab.stixsans-fallback` | STIX Sans as a fallback math font for missing glyphs. |

Style sheets compose from left to right. A later style can replace settings from an
earlier style. Use {ref}`how-to-plotting-figure-styles` to apply them in Python or Figure
Composer.

### External colormap libraries

The following optional libraries provide Matplotlib-compatible colormaps:

- [CMasher](https://github.com/1313e/CMasher)
- [cmocean](https://github.com/matplotlib/cmocean)
- [colorcet](https://github.com/holoviz/colorcet)
- [cmcrameri](https://github.com/callumrollo/cmcrameri)

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
