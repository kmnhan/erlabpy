# Data loader architecture

ARPES facilities use different raw-data formats. File grouping, scan identifiers,
coordinate names, and metadata also vary by endstation. ERLabPy keeps
endstation-specific file parsing and metadata handling in loaders. A loader reads data
from one acquisition system and returns xarray objects that use the conventions
described in {doc}`data-conventions`.

## Loader responsibilities

| Part | Function |
| --- | --- |
| Load call | Supplies a file path or scan identifier |
| Selected loader | Identifies associated files and reads their arrays and metadata |
| Loader post-processing | Renames coordinates and attributes, converts units, and applies endstation-specific processing |
| Return value | Provides a labeled `DataArray`, `Dataset`, or `DataTree` |

The loader does not make all endstations identical. It maps equivalent quantities to
common names and preserves additional source information as coordinates or attributes.

## Loader selection

The registry gives each loader a short name. The selected loader and the data directory
are separate settings.

| Setting | Controls | When it changes |
| --- | --- | --- |
| Default loader | File format and scan layout | Until changed, or only inside {func}`erlab.io.loader_context` |
| Data directory | Search location for relative paths and scan identifiers | Until changed, or only inside {func}`erlab.io.loader_context` |
| Loader arguments | Options for one source or scan | One load call |

{func}`erlab.io.set_loader` changes the default loader for later calls.
{func}`erlab.io.loader_context` temporarily sets the loader and, optionally, the data
directory inside its `with` block. ERLabPy restores both previous settings when the
block exits.

## Data across multiple files

Some acquisition systems store one scan step or detector frame in each file. The
loader identifies the related files and maps their metadata to scan coordinates. It
then combines the parts by physical coordinate order.

This differs from concatenating unrelated arrays by position. The loader's
{meth}`~erlab.io.dataloader.LoaderBase.identify` method determines which files belong
to the scan and supplies their scan coordinates.

## Loader extensions and plugins

| Option | Use it for | Do not use it for |
| --- | --- | --- |
| Loader extension | Renaming local metadata, adding logbook values, or turning attributes into coordinates | A new file reader or scan-grouping rule |
| Loader plugin | New file formats, source identification, scan grouping, or post-processing | Small changes to an existing loader |

For work with supported sources, see {doc}`../how-to/python/loading-and-saving` or
{doc}`../how-to/gui/loading-and-saving`. Loader authors can use
{doc}`../contributing/loaders`. The public loader interfaces are in
{mod}`erlab.io.dataloader`.
