# Data loading and saving

Use these guides to load data, correct its metadata, and save data from ImageTool. See
{ref}`ARPES data conventions <data-conventions>` for the coordinate and metadata names
required by ARPES-specific tools.

(how-to-gui-manager-open)=

## Loading data files into Manager

(imagetool-manager-start)=

### Starting the Manager

Run `itool-manager` in an environment that contains ERLabPy, or open the standalone
application. See {ref}`imagetool-manager-standalone` for standalone installation.

:::{note}
Changing bin widths for the first time after installation can take several minutes
while the application builds caches. Later launches are faster.
:::

(imagetool-manager-open)=

### Loading the files

1. Choose {menuselection}`File --> Add Data Files…`, or drag the files into the Manager.
2. Select the loader that matches the acquisition format.
3. If required, expand {guilabel}`Loader Extensions` and enter temporary load options.
4. Open the files and inspect the resulting dimensions, coordinates, and attributes in
   ImageTool.

To add or replace coordinates and attributes while loading files, use
{ref}`Acquisition Context <imagetool-manager-acquisition-context>`.

Use {guilabel}`Spreadsheet Metadata` when values must come from an Excel workbook or
public Google Sheet. See {ref}`io-spreadsheet-metadata` for matching rules.

:::{note}
For a scan recorded across several files, opening any member loads and concatenates the
complete scan when the selected loader supports it. Choose the loader suffixed with
{guilabel}`Single File` when only the selected file is required.
:::

Use {menuselection}`File --> Data Explorer` or {kbd}`Ctrl+E` when files must be previewed
before loading. Use {ref}`how-to-gui-open-data-in-imagetool` for data already present in
Python.

When loaded data has more than four effective dimensions, select or aggregate
dimensions in {guilabel}`Reduce Dimensions to Open` until the preview is a non-empty 2D,
3D, or 4D array.

(how-to-gui-export-data-from-imagetool)=

(imagetool-export)=

## Saving ImageTool data to a file

1. Activate the ImageTool that contains the data to save.
2. Choose {menuselection}`File --> Save As…`.
3. Select NetCDF, HDF5, or Igor Binary Wave (`.ibw`) from the file-type list.
4. Choose the output path and save the file.

Open the saved file with xarray or ImageTool and confirm that its dimensions,
coordinates, and attributes are present. Igor Binary Wave supports fewer dimensions and
metadata structures than HDF5 or NetCDF.

(how-to-gui-correct-manager-metadata)=

(imagetool-manager-acquisition-context)=

## Applying acquisition metadata while loading

Use Acquisition Context to add missing acquisition coordinates or attributes, or to
replace incorrect values from the data files. Some endstations store incomplete or
incorrect metadata.

Open {menuselection}`File --> Acquisition Context…` and enter the current values. You
can enter fields directly or use {guilabel}`Add from Selected ImageTool…` to copy them
from a selected ImageTool.

When the current context must be applied as you load new files, turn on
{guilabel}`Apply automatically when loading data from files` and save. Update the
context whenever acquisition conditions change.

Load one file and inspect its coordinates and attributes before applying the context to
the rest of the acquisition. To change data that is already open, use
{ref}`how-to-gui-correct-loaded-metadata`.

(how-to-gui-correct-loaded-metadata)=

(imagetool-manager-metadata-editor)=

## Correcting metadata after loading

Select one or more ImageTools and choose {menuselection}`Edit --> Metadata Editor…`.
Each row represents one ImageTool. Use {guilabel}`Fields` to choose which coordinates
and attributes appear as columns, then edit or paste values as you would in a logbook.

A marker in the upper-right corner shows that a value was assigned in ImageTool after
loading either from an acquisition context or manual assignment, rather than from the
file metadata. Use {guilabel}`Revert Assignment` to restore its earlier value.

Inspect the edited coordinates and attributes before momentum conversion or fitting.
Use {ref}`how-to-gui-correct-manager-metadata` when the same correction must be applied
while later files are loaded.
