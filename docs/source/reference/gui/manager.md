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

(imagetool-manager-tutorial)=

## Tutorial

Select {menuselection}`Help --> Tutorial` to start the Manager tutorial. Follow the
instructions in the application. See {ref}`manager-tutorial` for the launch steps.

## Keyboard shortcuts

Select {menuselection}`Help --> Keyboard Shortcuts` to open the searchable shortcut
reference. It includes shortcuts for Manager, ImageTool, Data Explorer, and Figure
Composer. The dialog shows the keys for your operating system.

(imagetool-manager-organize)=

## Data and tool tree

The left pane lists ImageTool windows, analysis tools, figures, and derived ImageTool
windows. Top-level ImageTools show an index and optional data name. A row produced by
another row appears below its source. Selecting a row fills the right pane with its
details, recorded steps, and preview.

The primary input history forms the main sequence of recorded steps. Other inputs
appear below the operation that uses them. {guilabel}`Open Input` opens recorded input
data. Reverting a main step restores its saved data when available and drops later
steps. Editing an earlier step repeats the later operations. Input labels show the
current ImageTool index, a source file name, {guilabel}`Saved`, or {guilabel}`Missing`.

The toolbar and row context menu provide these actions:

- {guilabel}`Show`, {guilabel}`Hide`, and {guilabel}`Remove` control the selected
  windows. Removing a row also removes its managed window.
- {guilabel}`Rename` changes displayed row names. {guilabel}`Duplicate` copies selected
  windows and their current state.
- {guilabel}`Arrange Selected Windows…` places selected windows in a grid.
- {guilabel}`Reset Index` renumbers top-level ImageTools from zero.
- {guilabel}`Link` and {guilabel}`Unlink` control shared cursors, slices, bins, and plot
  layout proportions. {guilabel}`Link` leaves color settings independent by default.
  Notebook and API calls keep color linking enabled by default.
- {guilabel}`Offload to Workspace` replaces in-memory data with a Dask-backed array from
  the saved workspace. {menuselection}`Dask --> Load Into Memory` in ImageTool reverses
  this operation.
- {guilabel}`Concatenate` calls {func}`xarray.concat` for selected ImageTool data and
  opens the result.
- {guilabel}`Open in ftool…` opens a weighted fit from two selected ImageTool rows.
  The dialog assigns one row to {guilabel}`Data` and one to
  {guilabel}`Standard uncertainty`; use {guilabel}`Swap` to exchange them. The data
  row remains the parent, and the uncertainty row remains a named `ftool` dependency.
  The uncertainty must align with and broadcast to the data. Its values must be finite
  and positive where the data is finite. If it is missing or incompatible, Manager
  does not open an unweighted `ftool`.
- {guilabel}`New Empty Figure` creates a Figure Composer window without data sources or
  recipe steps.
- {guilabel}`Add to Figure…` creates or updates a Figure Composer figure.
- {guilabel}`Reload Data` resolves current sources and repeats the recorded operations.
  Inputs captured by {guilabel}`Replace Current` retain their recorded data.
  The action reports missing files or inputs when replay is not possible. Stored Python
  remains paused until you review an unverified workspace.
- {guilabel}`Edit Note` and {guilabel}`Copy Note` manage the plain-text note stored with
  a workspace row.

Tree badges describe live state:

- A colored badge identifies linked ImageTools. Its tooltip reports whether color
  linking is on or off. Click the badge to open {guilabel}`Link Settings`. The
  {guilabel}`Link color settings` checkbox enables or disables color linking for the
  whole group. Select {guilabel}`OK` to apply the choice to subsequent changes without
  replacing current color settings, or {guilabel}`Cancel` to keep the previous choice.
  {guilabel}`Unlink This Window` removes only that window from the group. A saved
  workspace preserves each group's color linking setting.
- The Dask icon identifies chunked arrays.
- A variable-name badge identifies a watched notebook variable.
- {guilabel}`Stale`, {guilabel}`Unavailable`, and {guilabel}`Auto` describe update state
  for results that follow other ImageTools.
- {guilabel}`Changed` and {guilabel}`Missing` describe results that depend on several
  live ImageTools.

```{versionchanged} 3.28.0
Manager's {guilabel}`Link` action leaves color linking off by default. The link badge
opens settings for the group.
```

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

- {guilabel}`Open Child Window` places the result below its source.
- {guilabel}`Open Top-Level Window` places the result at the top level. Both choices
  record the transformation and its inputs.
- {guilabel}`Replace Current` replaces the data in the active ImageTool and retains
  the earlier data and recorded steps in its history.

For a result created from live ImageTools, expand {guilabel}`Inputs used` in the
{guilabel}`Provenance` panel to see the revisions that produced the displayed data.
Open a recorded input in a separate ImageTool to inspect its array. To change a
shared operation for this result alone, expand the input and edit the operation there.
For a stale result, expand {guilabel}`Inputs for next update` below the provenance
list to inspect its planned inputs. The Auto indicator shows whether the result
updates automatically. {guilabel}`Update Now` uses
current inputs and updates stale prerequisites. {guilabel}`Reload Data` rebuilds
followed inputs from their origins. Closing a source window retains its last revision
for results that still use it.

If you use {menuselection}`File --> Open` in a result window, the window keeps
its place in the tree. Its new input is the selected file. {guilabel}`Reload Data`
reads that file, and changes to the previous input no longer mark the window stale.

This also applies when you transform an ImageTool after reopening a workspace.
If the workspace retained the prepared result of a file load, Reload repeats
the saved load and preparation steps. If only the displayed array can be
identified, the preview shows that checkpoint. Reload the source ImageTool
first to replay any earlier steps that it owns.

```{versionchanged} 3.30.0
New child and top-level transform results can follow the same source. Moving a result
to the top level preserves its input bindings.
```

(imagetool-manager-workspace-properties)=

## Workspace properties

Open {menuselection}`File --> Workspace Properties` to inspect the active workspace.
The dialog shows its path, file type, size, modification time, stored-code status,
unsaved-change status, and number of open top-level windows.

(imagetool-manager-code-trust)=

### Stored code

A `.itws` workspace can contain Python used by recorded operations, Figure Composer
recipes, saved lmfit models or results, and parameter expressions. Manager restores
data, window state, and other non-executable content before it allows stored code to
run.

When stored code is paused:

| Stored content | Effect |
| --- | --- |
| Recorded Python operations | Operations that require Python do not replay. |
| Figure Composer Python and custom transforms | The preview can be incomplete, and export is unavailable for that incomplete figure. |
| Saved fitting code and parameter expressions | The related restoration, replay, or fitting operation does not run. |

The {guilabel}`Stored code` field has these values:

| Value | Meaning |
| --- | --- |
| `No stored executable content` | The workspace has no stored Python that requires review. |
| `Stored executable content is paused` | The workspace contains code that ERLab has not verified. |
| `Trusted by saved signature` | The executable content matches a workspace that you previously approved and saved. |
| `Trusted location` | The workspace is in a configured trusted folder. |
| `Trusted local document` | The executable content belongs to the current local workspace session. |

When review is available, the warning banner and Workspace Properties dialog show
{guilabel}`Review and Trust…`. Approval applies to all executable content listed in the
review dialog. If executable content in a saved workspace no longer matches its saved
approval, ERLab pauses it again. When you add or change code in an ERLab editor, ERLab
treats only that explicit edit as local work.

A standalone ImageTool does not replay stored Python from provenance. Open the original
workspace in Manager to review and replay it.

Use {ref}`how-to-gui-review-workspace-code` for the review procedure. The {ref}`Security
settings <options-trusted-workspace-folders>` describe trusted folders and saved
approvals.

## Data Explorer and Console

(imagetool-manager-data-explorer)=

### Data Explorer

Open the explorer from {menuselection}`File --> Data Explorer` or {kbd}`Ctrl+E`.

Use it when you want to browse folders, preview metadata, queue batch loads, and then
open selected files into the manager without writing code.

The explorer can also be launched standalone from Python or the command line for browsing
and previewing. Opening selected files into ImageTool analysis still requires a running
ImageTool manager, which is why launching it from the manager is the recommended path.

When launched from the manager, loader options are shared with the manager's
file-loading dialogs and across all Data Explorer tabs. This includes configured
spreadsheet metadata.

For supported general file formats and standalone launch paths, see
{ref}`guide-data-explorer`.

### Angle calculator

Open the calculator from {menuselection}`Apps --> Angle calculator`.
Use it to compare angular coverage with Brillouin-zone sections from a three-dimensional
lattice, model pockets, and a reference image. The Manager reuses the calculator window
and saves its state in the workspace. See {ref}`guide-angle-calculator`.

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
  # Get an independent DataArray snapshot of the first window
  tools[0].data

  # Inspect the child rows under the first window
  tools[0].children

  # Create an ImageTool containing the difference of the first two windows
  tools[0] - tools[1]

  # Use complicated expressions
  tools[0].qsel(alpha=slice(-1, 1)).qsel.average("eV")
  era.transform.rotate(
      tools[0],
      2.0,
      axes=("alpha", "eV"),
      center={"eV": 0.0, "alpha": 1.5},
      reshape=False,
      mode="nearest",
  )

  # Use a child ImageTool in a similar calculation
  tools[0].children[0] - tools[1]

  # Concatenate windows while retaining their Manager inputs
  xr.concat([tools[0], tools[1]], dim="scan")


  # Python functions can receive tool handles directly
  def normalize(data):
      return data / data.max()


  normalize(tools[0])

  # Keep the result in the console, then open it later
  diff = tools[0] - tools[1]
  diff.qshow(manager=True)

  # Replace data in the first window
  tools[0].data = tools[0].assign_coords(time=tools[1].time)
  ```

`tools[i].data` returns an independent, editable {class}`xarray.DataArray`
snapshot of the displayed data. The Manager does not track edits to that array
or its NumPy views. Assign the array back to apply your changes:

```python
a = tools[0].data
a[0, 0] = 100  # Changes a only
tools[0].data = a  # Replaces the window data
```

This assignment records a new checkpoint because the Manager does not know which
Python operations produced `a`. Existing results retain their recorded inputs.
To edit the window directly and record the operation, use `tools[0][0, 0] = 100`.
The handle `tools[0]` continues to use the current window data in expressions such
as `tools[0] - tools[1]`. A saved method such as `select = tools[0].isel` also uses
the current window data when you call `select(...)`.

A calculated result such as `diff = tools[0] - tools[1]` retains its calculated
values and the input revisions used to produce them. `diff.data` returns an
independent snapshot, just like `tools[i].data`. To modify a calculated result,
edit its snapshot and assign that array to a window. Opening `diff` itself
preserves its original values and recorded steps. Use
`diff.qshow(manager=True, replace=1)` to replace window 1 with that recorded
result while preserving the inputs used by earlier results.

Handles support arithmetic, comparisons, indexing, selection and reduction
methods, `qsel`, supported `kspace` operations, and named `coarsen` reductions.
Use `dir(handle)` to inspect the available operations. Shape and dimension reads
do not copy the full array. A coarsening operation that has already been
constructed retains its input even if the source window changes.

Use `.data` for ordinary xarray and NumPy operations, including coordinate and
attribute edits, callbacks, and raw array conversion. Access such as
`tools[0].values` or `tools[0].attrs` raises an error that explains this boundary.
For example:

```python
a = tools[0].data
a.coords["gain"].values[:] = 7
a.attrs["sample"] = "edited"
tools[0].data = a
```

The supplied `xr`, `era`, and `erlab` aliases support direct recorded calls such
as `xr.concat([tools[0], tools[1]], dim="scan")` and
`era.transform.rotate(tools[0], 5.0, reshape=False)`. ERLab transformations with
a recorded operation and multi-input `era.transform.shift` use the same path.
Calls with ordinary DataArrays keep normal library behavior. Unsupported tracked
calls require snapshots. Normal NumPy ufunc calls such as
`np.sin(tools[0])` retain provenance. Mutable output arguments such as `out=`
require ordinary snapshots.

Functions imported separately keep their normal behavior. Use `tools.apply`
to call one with handles, for example
`tools.apply(concat, [tools[0], tools[1]], dim="scan")` after
`from xarray import concat`. Importing `xarray as xr` replaces the supplied
console alias, so use `tools.apply` for that imported module as well.

Python functions receive the objects you pass to them. Passing a window handle
lets the function edit that window through the same operations used directly in
the console:

```python
def set_first_value(data):
    data[0, 0] = 100


set_first_value(tools[0])  # Edits the window and records the operation
set_first_value(tools[0].data)  # Edits only the independent snapshot
```

The `.data` snapshot boundary also applies inside functions. Assign an edited
DataArray to the handle's `.data` property to apply coordinate or metadata
changes. An edit already applied to a window is retained if a later statement
raises an exception. A failed edit leaves the window and its provenance unchanged.

The recorded result contains the array operations that the function performed.
Refresh replays those operations; it does not rerun arbitrary Python control
flow. For example, if a Python conditional chooses a multiplier of 2, that
multiplier remains 2 during Refresh even if the updated source would take another
branch. Python scalars obtained through `.item()` are fixed values in subsequent
recorded calculations.

Python augmented assignment follows the same boundary. `tools[0].data += 1`
reads a snapshot and assigns it back as a new checkpoint. In contrast,
`h = tools[0]; h += 1` assigns a calculated result to the variable `h` and leaves
the window unchanged.

:::{versionchanged} 3.30.0
`.data` returns a snapshot for both window handles and calculated results. An
expression such as `tools[0].data[0, 0] = 100` changes only that temporary snapshot.
Use `tools[0][0, 0] = 100` to edit the window. Saved window methods and accessors
use the current data when called. Console functions receive live handles directly.
Mutable exports and unsupported calls require `.data`. Supported library calls
retain provenance through the supplied aliases or `tools.apply`.
:::

:::{tip}
Drag one row from the {guilabel}`Data/Tools` tree into the console to insert its
`tools[...]` expression where you drop it. Nested rows insert their full
`.children[...]` path automatically.
:::

Run standard Python, `%magic` commands, or inspect objects with `?` exactly as you would
in a notebook.

For notebook integration procedures, see {ref}`working-with-notebooks`.

(imagetool-manager-notebook-synchronization)=

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
