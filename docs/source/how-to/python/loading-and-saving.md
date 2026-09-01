# Data loading and saving

Use these guides to load experimental data, add missing acquisition metadata, find
scans, and save results.

(loading-arpes-data)=

(how-to-python-configure-loader)=

## Loading data from a supported endstation

Select the loader for your endstation:

{{ available_loader_table }}

Loader names are case-sensitive. Check the registry in your installed environment
because the available loaders depend on the ERLabPy version and installed plugins:

```python
import erlab

erlab.io.loaders
```

Replace the loader name, data directory, and identifier with values for the experiment.
The identifier can be a scan number or a file name when the selected loader supports
that form:

```python
identifier = 42
with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    data = erlab.io.load(identifier)
```

Inspect the result before analysis:

```python
data
data.coords
data.attrs
```

Confirm that the dimensions, coordinates, and acquisition metadata match the
measurement. Before momentum conversion or Fermi edge fitting, also confirm that the
result follows the {ref}`data-conventions`.

Keep the context open when you must load several scans from the same experiment:

```python
with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    data_1 = erlab.io.load(1)
    data_2 = erlab.io.load(2)
```

If ERLabPy cannot find the scan, confirm the selected loader, the accepted identifier
form, and the data directory. If the endstation is not listed, follow the
procedure in {doc}`../../contributing/loaders` instead of forcing the file through a
different loader. See {doc}`../../explanation/loader-architecture` for how loaders
standardize endstation data.

(how-to-python-load-several-experiments)=

## Loading data from several experiments in one notebook

Use a separate {func}`erlab.io.loader_context` for each experiment. The context limits
the selected loader and data directory to its `with` block:

```python
import erlab

with erlab.io.loader_context("merlin", data_dir="/data/merlin-experiment"):
    merlin_map = erlab.io.load(42)

with erlab.io.loader_context("i05", data_dir="/data/i05-experiment"):
    i05_map = erlab.io.load(17)
```

Use the same pattern when two experiments use the same loader but have different data
directories. Give each result a name that identifies its source. Then inspect the
coordinates and attributes before you compare or combine the data:

```python
merlin_map.coords
merlin_map.attrs

i05_map.coords
i05_map.attrs
```

Do not depend on the order of repeated {func}`erlab.io.set_loader` or
{func}`erlab.io.set_data_dir` calls. Each call changes the active global setting. A
loader context keeps each load operation next to the settings that control it.

(how-to-python-load-igor)=

## Loading data exported from Igor Pro

Load a single `.ibw` wave or a single-wave `.itx` file as a
{class}`DataArray <xarray.DataArray>`:

```python
import xarray as xr

data = xr.load_dataarray("/path/to/wave.ibw")
```

For a packed experiment (`.pxp` or `.pxt`) that contains several folders or waves,
open the hierarchy as a {class}`DataTree <xarray.DataTree>` and select the required
node:

```python
with xr.open_datatree("/path/to/experiment.pxp") as experiment:
    print(experiment.groups)
    wave_node = experiment["/folder/wave_name"]
    data = wave_node.dataset["wave_name"].load()
```

Use a group path listed in {attr}`groups <xarray.DataTree.groups>`. Then select the wave
variable from that node. The call to {meth}`load <xarray.DataArray.load>` reads the
selected wave before the file closes.

For an HDF5 file exported by Igor Pro, select the ERLabPy backend explicitly:

```python
data = xr.load_dataset("/path/to/export.h5", engine="erlab-igor")
```

Inspect the dimensions, coordinates, and attributes after loading. If a complex packed
experiment does not load correctly, export the required wave as `.ibw` and load that
file. See {class}`erlab.io.igor.IgorBackendEntrypoint` for the supported Igor formats
and backend behavior.

(how-to-python-load-multifile-scan)=

## Loading a scan stored across multiple files

Use this guide when one scan is stored in files such as `f_003_S001.pxt`,
`f_003_S002.pxt`, and later sequence files. Select the loader and directory for the
experiment:

```python
import erlab

loader = erlab.io.loaders["merlin"]
data_dir = "/path/to/data"
```

Load and concatenate the complete scan with its scan number or with any file in the
scan:

```python
scan = loader.load(3, data_dir=data_dir)
scan_from_first_file = loader.load("f_003_S001.pxt", data_dir=data_dir)
scan_from_second_file = loader.load("f_003_S002.pxt", data_dir=data_dir)
```

Each call returns the same concatenated data when the selected loader supports
multi-file scans.

To load only one file, set `single=True`:

```python
single_file = loader.load("f_003_S001.pxt", data_dir=data_dir, single=True)
```

To inspect the scan as separate arrays, disable concatenation:

```python
separate_files = loader.load(3, data_dir=data_dir, combine=False)
```

If loading one sequence file does not find its companions, confirm that the active
loader supports multi-file scans and that every file is in the selected data directory.
See {meth}`erlab.io.dataloader.LoaderBase.load` for the loader arguments.

(io-spreadsheet-metadata)=

(how-to-python-add-spreadsheet-metadata)=

## Adding logbook metadata while loading data

Use an Excel log when acquisition settings are not stored in each data file. Identify
the column that matches each file and map the remaining columns to ERLabPy coordinates
or attributes:

```python
import erlab
from erlab.io.metadata import ExcelMetadataSource

metadata = ExcelMetadataSource(
    "acquisition-log.xlsx",
    sheet_name="Measurements",
    file_name_column="File",
    coordinate_mapping={
        "Photon Energy": "hv",
        "Temperature": "sample_temp",
    },
    attribute_mapping={"Polarization": "polarization"},
    overwrite=False,
)

with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    data = erlab.io.load(42, metadata=metadata)
```

Set `overwrite=True` only when the logbook values must replace scalar values already
stored in the file. Confirm the matched row and the resulting coordinates before using
the data for momentum conversion or fitting.

If a nonstandard file name cannot be matched to its row, supply the logbook file number
explicitly:

```python
with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    data = erlab.io.load("custom-name.pxt", metadata=metadata, file_number=42)
```

For a public Google Sheet, use
{class}`GoogleSheetsMetadataSource <erlab.io.metadata.GoogleSheetsMetadataSource>` with
the same mappings. Give anyone with the link view access. Use `row_range=` when one
sheet contains repeated file numbers from several experiments.

See {class}`erlab.io.metadata.ExcelMetadataSource` and
{class}`erlab.io.metadata.GoogleSheetsMetadataSource` for matching and range syntax.

(io-loader-extensions)=

(how-to-python-preserve-file-metadata)=

## Preserving file metadata as coordinates

Use a temporary loader extension when metadata needed during concatenation is stored as
a file attribute. For one load, promote the attribute through `loader_extensions`:

```python
import erlab

with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    data = erlab.io.load(
        1,
        loader_extensions={"coordinate_attrs": ("scan_number",)},
    )
```

For several loads, apply the extension only inside a context manager:

```python
with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    with erlab.io.extend_loader(coordinate_attrs=("scan_number",)):
        data_1 = erlab.io.load(1)
        data_2 = erlab.io.load(2)
```

Inspect the loaded coordinates before analysis. The metadata key must exist in the
files handled by the active loader.

See {func}`erlab.io.extend_loader` for the extension fields and their accepted values.

(io-summarizing-data)=

(how-to-python-summarize-data-directory)=

## Summarizing available scans in Python

Select the loader, then request a table for the data directory:

```python
import erlab

with erlab.io.loader_context("merlin", data_dir="/path/to/data"):
    summary = erlab.io.summarize()
```

Filter the returned {class}`DataFrame <pandas.DataFrame>` to find scans that match the
required acquisition conditions. If no path is supplied, ERLabPy uses the current data
directory.

Use the {ref}`Data Explorer <imagetool-manager-data-explorer>` instead when the goal is
interactive browsing, metadata preview, and opening selected files in Manager.

(how-to-python-save-data)=

## Saving analysis data with coordinates and metadata

Save an xarray {class}`DataArray <xarray.DataArray>` in an HDF5-backed NetCDF file when
dimensions, coordinates, and attributes must remain available to Python:

```python
data.to_netcdf("analysis-result.h5", engine="h5netcdf")
```

Open the file later with xarray:

```python
import xarray as xr

restored = xr.load_dataarray("analysis-result.h5", engine="h5netcdf")
```

Use {func}`erlab.io.igor.save_wave` only when the recipient requires an Igor Binary Wave
and the {class}`DataArray <xarray.DataArray>` satisfies the format limits:

```python
import erlab

erlab.io.igor.save_wave(data, "analysis-result.ibw")
```

An Igor Binary Wave supports at most four uniformly sampled dimensions and does not
preserve non-dimensional coordinates.
