# Python integration

Use these guides to open Python data in ImageTool and move data or operations between
Python, ImageTool, and Manager.
Read {doc}`Python and GUI workflows <../../explanation/python-and-gui-workflows>` to
understand when to move an operation between the two environments.

(how-to-gui-open-data-in-imagetool)=

## Opening Python data in ImageTool

Call ImageTool with the {class}`DataArray <xarray.DataArray>` from the current Python
session:

```python
import erlab.interactive as eri

eri.itool(data)
```

Pass a list and set `link=True` when several arrays must open with synchronized cursors
and bins. If the input has more than four effective dimensions, select or aggregate
dimensions in {guilabel}`Reduce Dimensions to Open` before opening the result.

See {ref}`imagetool-entry-points` for all Python, IPython, and VS Code entry points. Use
{ref}`how-to-gui-watch-notebook-variables` when changes must stay synchronized in both
directions.

(how-to-gui-load-imagetool-magic)=

(imagetool-ipython)=

## Loading ImageTool commands when a VS Code notebook starts

Use this task for notebooks that run in VS Code with the Jupyter extension.

1. Open the VS Code user or workspace `settings.json` file.
2. Add this setting:

   ```json
   "jupyter.runStartupCommands": [
       "%load_ext erlab.interactive"
   ]
   ```

3. Restart the notebook kernel.
4. Run `%itool --help` to confirm that the extension loaded.

This setting applies only to notebooks that VS Code starts. In another notebook
frontend, run `%load_ext erlab.interactive` in the notebook unless that frontend has an
equivalent startup-command setting.

(how-to-gui-copy-imagetool-code)=

(imagetool-round-trip)=

## Continuing an ImageTool operation in Python

1. Open Python data with {meth}`xarray.DataArray.qshow`,
   {func}`erlab.interactive.imagetool.itool`, or `%itool`.
2. Make the required selection or transformation in ImageTool.
3. Choose {guilabel}`Copy selection code` from a plot or {guilabel}`Copy Code` from the
   operation dialog.
4. Paste and run the copied expression in the notebook.

When ImageTool is managed, select the result row and use {guilabel}`Copy Full Code` to
include the recorded input and preceding operations. Inspect file paths, variable names,
and parameter values before running copied code.

Use {ref}`how-to-gui-watch-notebook-variables` instead when the ImageTool result must
update a live notebook variable. See {ref}`workflow-bridge-operations` for the
corresponding Python operations.

(how-to-gui-watch-notebook-variables)=

(working-with-notebooks)=
(imagetool-manager-watching)=

## Synchronizing a notebook variable with ImageTool

Load the IPython extension and start a watch:

```python
%load_ext erlab.interactive
%watch my_data
```

The Manager creates or reconnects an ImageTool row labeled `my_data`. Reassigning a
{class}`DataArray <xarray.DataArray>` to `my_data` updates the row after the notebook
cell finishes. Compatible ImageTool edits update the notebook variable.

Run `%watch my_data` again to force a refresh. To stop synchronization but keep the
ImageTool row, run:

```python
%watch -d my_data
```

Use `%watch -x my_data` when the row must also close. See
{ref}`the Manager command reference <imagetool-manager>` for all `%watch` forms.

If the variable is deleted or replaced by an object that is not a
{class}`DataArray <xarray.DataArray>`, the Manager breaks the watch and keeps a regular
ImageTool row.

(imagetool-manager-reconnect-watches)=

### Reconnecting variables after restarting

1. Open the saved Manager workspace.
2. Run the notebook cells that recreate the watched
   {class}`DataArray <xarray.DataArray>` variables.
3. Reconnect all matching names:

   ```python
   %watch --restore
   ```

Rows with missing variables or variables that are not
{class}`DataArray <xarray.DataArray>` objects remain disconnected. If several rows use
the same variable name, remove the unwanted watch before reconnecting so the Manager
does not have to guess.

When sharing the workflow, send both the notebook and `.itws` workspace. The files do
not need to be in the same directory, but the notebook must recreate the variable names
stored in the workspace.

### Using a non-IPython environment

Call the Python API with the namespace that contains the variable:

```python
from erlab.interactive.imagetool.manager import watch

watch("my_data", namespace=globals(), poll_interval_s=0.5)
```

Use {func}`watch <erlab.interactive.imagetool.manager.watch>` as
`watch("my_data", stop=True)` to stop one watch. Use `watch(stop_all=True)` to stop all
watches. Use `watch(restore=True)` to reconnect rows from the open workspace.

Provide `namespace=` when caller scope is not obvious, such as inside a helper or
callback. Use {func}`shutdown <erlab.interactive.imagetool.manager.shutdown>` before the
host application exits.

Use {ref}`how-to-gui-load-imagetool-magic` to load `%watch` automatically in notebooks.

(how-to-gui-copy-manager-data-to-python)=

(imagetool-manager-fetch)=

## Copying Manager data into Python

Use {func}`fetch <erlab.interactive.imagetool.manager.fetch>` inside a notebook or script to copy data out of the manager:

```python
from erlab.interactive.imagetool.manager import fetch

data = fetch(0)  # returns an xarray.DataArray copy
```

Because `fetch` returns a copy, you can safely modify it without touching the live window.

Use the row index shown by the Manager. When several Manager windows are running,
select the target Manager first as described in
{ref}`how-to-gui-select-manager-instance`.

(how-to-gui-transfer-data-between-notebooks)=

## Transferring Manager data between notebooks

Use IPython `%store` when data in one Manager session must be available to another
notebook kernel. The Manager console and the receiving kernel must use the same IPython
profile. They must also run as the same operating-system user.

1. Select the required ImageTool rows in the Manager.
2. Choose {menuselection}`File --> Store with IPython` or the matching row context-menu
   action.
3. Record the variable names used by the Manager.
4. In the receiving notebook, restore each variable:

   ```python
   %store -r my_data
   ```

5. Confirm that the restored object is a {class}`DataArray <xarray.DataArray>`:

   ```python
   import xarray as xr

   isinstance(my_data, xr.DataArray)
   ```

   The result must be `True` before you continue the analysis.

The restored {class}`DataArray <xarray.DataArray>` is independent of the live Manager
row. Use
{ref}`how-to-gui-watch-notebook-variables` instead when changes must remain synchronized
in both directions.

If `%store` cannot find the variable or the restored object is not a
{class}`DataArray <xarray.DataArray>`, use a file instead. Different IPython profiles
can cause this problem. Save the ImageTool data as NetCDF or HDF5 with
{ref}`how-to-gui-export-data-from-imagetool`. Then load the file in the receiving
notebook:

```python
import xarray as xr

my_data = xr.load_dataarray("my-data.h5", engine="h5netcdf")
```

Use the saved `.nc` path instead when you selected NetCDF. Inspect the restored
dimensions, coordinates, and attributes before you continue.

(how-to-gui-select-manager-instance)=

(imagetool-manager-multiple-instances)=

## Sending data to a specific Manager window

Multiple ImageTool Manager windows can run at the same time. The first live window has
index `0`. Later windows receive 0-based indexes in their start order.

1. Start the required Manager windows.
2. Inspect the live windows from the Python process that contains the data:

   ```python
   import erlab.interactive.imagetool.manager as itm

   itm.managers
   ```

3. Send the data with the index shown for the target Manager:

   ```python
   itm.managers[1].show(data)
   ```

4. Confirm that the new ImageTool row appears in the target Manager.

If more than one manager is running and no default has been selected, calls that use
`manager=True` raise an error instead of guessing. See
{ref}`imagetool-manager-selection` for default selection, explicit target arguments, and
IPython magic forms.
