# Data formats and loader plugins

## Data loader plugins

The {mod}`erlab.io` API contains the loader registry and common loading functions. The
following modules describe the available loading interfaces:

| Interface | Contents |
| --- | --- |
| {mod}`erlab.io.plugins` | Built-in data loader plugins |
| {mod}`erlab.io.dataloader` | Loader registry, plugin base class, loading behavior, and scan grouping |
| {mod}`erlab.io.igor` | Igor Pro xarray backend and export functions |
| {mod}`erlab.io.metadata` | Spreadsheet-backed metadata sources |
| {mod}`erlab.io.fitsutils` | FITS conversion functions |
| {mod}`erlab.io.nexusutils` | NeXus conversion functions |

Use {doc}`../how-to/python/loading-and-saving` for procedures on experimental files. See
{ref}`data-conventions` for the coordinate and metadata conventions required by
ARPES-specific tools.

## Generic file formats

When no data loader plugin applies, use the following instructions for file formats to
inspect its contents.

| Data | Interface |
| --- | --- |
| NetCDF and supported HDF5 files | {func}`xarray.open_dataarray`, {func}`xarray.open_dataset`, or {func}`xarray.open_datatree` |
| HDF5 files with an unknown group structure | {func}`xarray.open_groups` |
| CSV and Excel tables | {func}`pandas.read_csv` or {func}`pandas.read_excel`, followed by {meth}`pandas.DataFrame.to_xarray` when an xarray object is required |
| FITS data | {func}`erlab.io.fitsutils.fits_to_xarray` |
| NeXus data that requires explicit conversion | {mod}`erlab.io.nexusutils` |

An xarray object must follow the {ref}`ARPES data conventions <data-conventions>` before
it is used with an ARPES-specific analysis routine. Use {doc}`../contributing/loaders`
to add a new acquisition format or scan-grouping rule.
