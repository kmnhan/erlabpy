(imagetool-manager)=

# ImageTool Manager

```{image} ../../images/manager_light.png
:align: center
:alt: ImageToolManager window screenshot
:class: only-light
:width: 600px
```

:::{only} format_html

```{image} ../../images/manager_dark.png
:align: center
:alt: ImageToolManager window screenshot
:class: only-dark
:width: 600px
```

:::

{class}`ImageToolManager <erlab.interactive.imagetool.manager.ImageToolManager>` is an
application for managing multiple ImageTool windows, analysis tools, and Matplotlib
figures in one place. It is designed to keep your workflow organized when you are
working with many windows at once. It also provides saving and loading sessions,
synchronization with Jupyter notebooks, and a built-in IPython console for quick
calculations and data exploration.

(imagetool-manager-overview)=

## Why use the manager?

- Launch and watch many ImageTool windows simultaneously without interrupting your
  notebook or script.
- Keep nested ImageTool windows organized in a tree that shows their relationships and
  provenance.
- Update tools and ImageTool windows automatically when the ImageTool or tool that
  created them changes.
- Link multiple ImageTools, duplicate them, or update their data in place in case of
  real-time data acquisition.
- Save multiple windows and full hierarchies to a file, share them with collaborators,
  and reload them later to pick up right where you left off.
- Keep track of the code and steps that led to the data in each ImageTool window.
- Integration with Jupyter notebooks through the `%watch` magic, which creates windows
  that stay synchronized with notebook variables.
- Create Matplotlib figures from ImageTool data without writing code using the built-in
  {ref}`Figure Composer <figure-composer>`.
- Add lab-specific analysis routines and file loaders with Python extension scripts.
- Drag-and-drop files to open them quickly, or use the integrated data explorer to
  browse preview data.

For startup and file-opening procedures, see {ref}`how-to-gui-manager-open`.

For metadata procedures, see {ref}`how-to-gui-correct-manager-metadata`.

For workspace procedures, see {ref}`how-to-gui-save-manager-workspace`. For
derived-result procedures, see {ref}`how-to-gui-update-derived-results`.

For extension controls and states, see {ref}`imagetool-manager-extensions`. For
extension procedures, see {ref}`how-to-gui-manager-extensions`.

(imagetool-manager-organize)=

## Data and tool tree

The left pane lists ImageTool windows, analysis tools, figures, and derived ImageTool
windows. Top-level ImageTools show an index and optional data name. A row produced by
another row appears below its source. Selecting a row fills the right pane with its
details, recorded steps, and preview.

The toolbar and row context menu provide these actions:

- {guilabel}`Show`, {guilabel}`Hide`, and {guilabel}`Remove` control the selected
  windows. Removing a row also removes its managed window.
- {guilabel}`Rename` changes displayed row names. {guilabel}`Duplicate` copies selected
  windows and their current state.
- {guilabel}`Arrange Selected Windows…` places selected windows in a grid.
- {guilabel}`Reset Index` renumbers top-level ImageTools from zero.
- {guilabel}`Link` and {guilabel}`Unlink` control shared cursors, slices, bins, and plot
  layout proportions.
- {guilabel}`Offload to Workspace` replaces in-memory data with a Dask-backed array from
  the saved workspace. {menuselection}`Dask --> Load Into Memory` in ImageTool reverses
  this operation.
- {guilabel}`Concatenate` calls {func}`xarray.concat` for selected ImageTool data and
  opens the result.
- {guilabel}`New Empty Figure` creates a Figure Composer window without data sources or
  recipe steps.
- {guilabel}`Add to Figure…` creates or updates a Figure Composer figure.
- {guilabel}`Reload Data` reloads file-backed data and repeats its recorded operations.
  The action reports missing files or inputs when replay is not possible.
- {guilabel}`Edit Note` and {guilabel}`Copy Note` manage the plain-text note stored with
  a workspace row.

Tree badges describe live state:

- A colored badge identifies linked ImageTools.
- The Dask icon identifies chunked arrays.
- A variable-name badge identifies a watched notebook variable.
- {guilabel}`Stale`, {guilabel}`Unavailable`, and {guilabel}`Auto` describe update state
  for child results.
- {guilabel}`Changed` and {guilabel}`Missing` describe results that depend on several
  live ImageTools.

Enable {menuselection}`View --> Preview on Hover` to show row previews while moving the
pointer over the tree.

(imagetool-manager-nested-results)=

### Parent and child rows

An ImageTool or analysis tool opened from another managed row appears below the row that
created it. The child stores the source selection or operation needed to reproduce its
data. Compatible source changes can mark the child as stale or update it automatically.

(imagetool-manager-result-placement)=

### Result placement

ImageTool transformation dialogs use {guilabel}`Result Placement`:

- {guilabel}`Open Child Window` preserves the source and records a child result.
- {guilabel}`Open Top-Level Window` preserves the source but creates an independent
  top-level row.
- {guilabel}`Replace Current` replaces the data in the active ImageTool.

## Data Explorer and Console

(imagetool-manager-data-explorer)=

### Data Explorer

Open the explorer from {menuselection}`File --> Data Explorer` or {kbd}`Ctrl+E`.

Use it when you want to browse folders, preview metadata, queue batch loads, and then
open selected files into the manager without writing code. For most day-to-day browsing
it is faster than the interactive summary table in the I/O guide. Use
{func}`erlab.io.summarize` instead when you want the overview as a DataFrame in Python
or when you are developing loaders.

The explorer can also be launched standalone from Python or the command line for browsing
and previewing. Opening selected files into ImageTool analysis still requires a running
ImageTool manager, which is why launching it from the manager is the recommended path.

When launched from the manager, loader options are shared with the manager's
file-loading dialogs and across all Data Explorer tabs. This includes configured
spreadsheet metadata.

For the standalone tool page, see {ref}`guide-data-explorer`.

### Periodic Table

Open the periodic table from {menuselection}`Apps --> Periodic Table` or {kbd}`Ctrl+Shift+P`.

Use it when you want quick reference for core-level energies photoionization cross
sections.

For the standalone tool page, see {ref}`guide-ptable`.

### Console

For quick calculations and data exploration without leaving the manager, the embedded
IPython console is useful.

Toggle the embedded IPython console with {kbd}`Ctrl+J` or via the {guilabel}`View` menu.
The console exposes a `tools` list containing a provenance-aware handle for every
ImageTool. These handles are not {class}`xarray.DataArray` objects, but they support
many of the same operations and keep track of the manager history. For example:

  ```python
  # Access the underlying DataArray of the first window
  tools[0].data

  # Inspect the child rows under the first window
  tools[0].children

  # Create an ImageTool containing the difference of the first two windows
  tools[0] - tools[1]

  # Use complicated expressions
  tools[0].qsel(alpha=slice(-1, 1)).qsel.average("eV")
  era.transform.rotate(tools[0], 2.0, axes=("alpha", "eV"), reshape=False)

  # Use a child ImageTool in a similar calculation
  tools[0].children[0] - tools[1]

  # xarray module calls also keep manager inputs when they receive tool handles
  xr.concat([tools[0], tools[1]], dim="scan")


  # Simple helper functions defined in the console can receive tool handles directly
  def normalize(data):
      return data / data.max()


  normalize(tools[0])

  # Keep the result in the console, then open it later
  diff = tools[0] - tools[1]
  diff.qshow(manager=True)

  # Replace data in the first window
  tools[0].data = tools[0].assign_coords(time=tools[1].time)
  ```

:::{tip}
Drag one row from the {guilabel}`Data/Tools` tree into the console to insert its
`tools[...]` expression where you drop it. Nested rows insert their full
`.children[...]` path automatically.
:::

Run standard Python, `%magic` commands, or inspect objects with `?` exactly as you would
in a notebook.

For notebook integration procedures, see {ref}`working-with-notebooks`.

### Notebook synchronization commands

The `%watch` IPython magic connects DataArray variables to managed ImageTools. Its main
forms are:

```text
%watch data1 data2       Start or refresh watches
%watch                   List watched names
%watch --restore         Reconnect saved rows by variable name
%watch -d data1          Stop watching and keep the row
%watch -x data1          Stop watching and close the row
%watch -z                Stop all watches
%watch -xz               Stop all watches and close their rows
```

The public {func}`watch <erlab.interactive.imagetool.manager.watch>` function provides
the same operations outside IPython. Non-IPython environments use polling. The
`poll_interval_s` argument controls the interval. Use
{func}`maybe_push <erlab.interactive.imagetool.manager.maybe_push>` for an immediate
check and {func}`shutdown <erlab.interactive.imagetool.manager.shutdown>` to stop watcher
threads.

(imagetool-manager-automation)=

## Automation APIs

If you wish to integrate the manager into custom workflows, you can programmatically load data and control ImageTool windows in the manager. Use the public functions exported from {mod}`erlab.interactive.imagetool.manager`:

```python
from erlab.interactive.imagetool.manager import (
    load_in_manager,
    replace_data,
    show_in_manager,
)

# Open raw files and let the manager choose the loader interactively
load_in_manager(["scan1.pxt", "scan2.pxt"])

# Open raw files with temporary loader extensions
load_in_manager(
    ["scan1.pxt", "scan2.pxt"],
    "merlin",
    loader_extensions={"coordinate_attrs": ("scan_number",)},
)

# Open two ImageTools and link their cursors
show_in_manager([data_a, data_b], link=True, target=1)

# Replace the dataset at index 3 with a new result
replace_data(3, new_data, target=1)
```

Additional functions and objects such as {data}`managers <erlab.interactive.imagetool.manager.managers>`, {func}`replace_data <erlab.interactive.imagetool.manager.replace_data>`, {func}`watch <erlab.interactive.imagetool.manager.watch>`, and {func}`manager_selection_info <erlab.interactive.imagetool.manager.manager_selection_info>` give you finer control when building custom acquisition pipelines or editor integrations.

These functions use ZeroMQ to communicate with the GUI. The manager uses a user-specific
live registry for discovery. Normal routing is for Python processes in the same user
session as the manager. See the API documentation for details.

(imagetool-manager-selection)=

### Manager selection

Manager indexes are 0-based. The {data}`manager registry
<erlab.interactive.imagetool.manager.managers>` lists live Manager windows and their
indexes.

ERLabPy selects the only live Manager automatically. It also uses the default Manager
that the current Python process selected. An operation raises an error when several
Manager windows are live and neither rule selects one.

The public interfaces provide these selection forms:

- A registry handle provides `show`, `load`, `replace`, `fetch`, and `watch` methods for
  one index. Its `use` method sets the process default.
- Manager functions such as {func}`show_in_manager
  <erlab.interactive.imagetool.manager.show_in_manager>` accept a `target` index.
- {meth}`xarray.DataArray.qshow` accepts a Manager index in `manager`.
- `%itool` and `%watch` accept a Manager index with `-m`.
- `%manager list`, `%manager use INDEX`, `%manager current`, and `%manager clear` inspect
  or change the process default.

Use {ref}`how-to-gui-select-manager-instance` for one task sequence.

For standalone installation procedures, see {ref}`imagetool-manager-standalone`.
