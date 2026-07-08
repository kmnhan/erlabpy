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
- Drag-and-drop files to open them quickly, or use the integrated data explorer to
  browse preview data.

(imagetool-manager-start)=

## Starting the manager

- If you have a Python environment with ERLabPy installed:

  Run `itool-manager` in a terminal or command prompt window in the environment where
  ERLabPy is installed.

- You can also install the manager as an application on your operating system without
  installing Python. See {ref}`imagetool-manager-standalone` for instructions.

:::{note}

Opening an ImageTool window and changing the bin widths for the very first time after
installing may take a couple of minutes as caches are built. Subsequent launches will be
much faster.

:::

(imagetool-manager-open)=

## Opening and replacing ImageTool windows

Once the manager is running, you can open ImageTools in several ways:

- {meth}`xarray.DataArray.qshow` or {func}`erlab.interactive.imagetool.itool` with
  `manager=True` sends windows to the manager:

  ```python
  data.qshow(manager=True)
  eri.itool([d1, d2], manager=True, replace=[1, 2])
  ```

  Pass `replace=` to update data in existing windows instead of creating new ones.

  :::{tip}

  For working with multiple manager instances, see
  {ref}`imagetool-manager-multiple-instances`.

  :::

- {ref}`ImageTool’s %itool magic command <imagetool-entry-points>` with the `--manager`
  (or `-m`) flag in an IPython session or Jupyter notebook.

  ```python
  %itool -m darr
  %itool -m 1 darr
  ```

- The {menuselection}`File --> Move to Manager` ({kbd}`Ctrl+Shift+M`) action from an ImageTool
  window opened outside the manager. This action moves the active ImageTool to the
  manager.

- Use the manager’s {menuselection}`File --> Add Data Files…` action to load data in a new
  ImageTool.

- Drag and drop supported ARPES data into the manager window.

  In the dialog that appears, you can choose the plugin to use for loading the data. For
  data loader plugins loaders, expand {guilabel}`Loader Extensions` to set custom load
  arguments, including {func}`erlab.io.extend_loader` options. The {guilabel}`name_map`
  and {guilabel}`coordinate_attrs` rows have buttons that inspect the first selected
  file and help build the literal values interactively.

  :::{hint}

  For scans that are recorded across multiple files, drag and dropping any file in the
  scan will automatically load and concatenate the entire scan. If you want to load only
  the file you dropped, choose the plugin suffixed with {guilabel}`Single File` in the
  dialog.

  :::

- Launch the built-in data explorer from {menuselection}`File --> Data Explorer` or
  {kbd}`Ctrl+E` when you want directory browsing and metadata preview before opening
  selected files in the manager. Use the loader options button next to the loader
  selector to apply the same `loader_extensions=` settings when opening selected files.

- Watch notebook variables with the `%watch` magic to create windows that stay
  synchronized with your data structures. Use `%watch -m 1 darr` to watch into manager
  `#1`. See {ref}`working-with-notebooks`.

  :::{tip}

  This is the recommended way when you are working with notebooks, because it keeps your
  workflow connected to your code and automatically synchronizes changes in both
  directions.

  :::

- For custom integration with other workflows, scripts can call
  {func}`erlab.interactive.imagetool.manager.show_in_manager` or
  {func}`erlab.interactive.imagetool.manager.load_in_manager` directly (see
  {ref}`imagetool-manager-automation`).

Manual manager opening paths, including `manager=True`, `%itool -m`, file loading,
drag-and-drop, Data Explorer, `show_in_manager`, and `load_in_manager`, use the same
ImageTool input preparation as a standalone ImageTool window. When opened data has more
than four effective dimensions, the manager shows {guilabel}`Reduce Dimensions to Open`
before adding the ImageTool row. Select or aggregate dimensions there until the preview
shows a non-empty 2D, 3D, or 4D result, or cancel to skip opening that data.

(imagetool-manager-multiple-instances)=

## Multiple manager instances

Multiple ImageTool Manager windows can run at the same time. The first live manager is
manager `#0`, and later managers receive 0-based indexes in the order they start.

To start a new manager instance, choose {menuselection}`File --> New Manager Window` from an
existing manager window.

When more than one manager is running, either pass the index like `manager=2` or set a
default for the current Python process or notebook kernel:

```python
import erlab.interactive.imagetool.manager as itm

itm.managers
itm.managers[1].use()
itm.managers[1].show(data)
data.qshow(manager=True)
other.qshow(manager=0)
%itool -m 1 data
%watch -m 1 data
```

The default is stored in the current session. The same actions are also available as
IPython magics `%manager list`, `%manager use 1`, `%manager current`, and `%manager
clear`.

If more than one manager is running and no default has been selected, calls that use
`manager=True` raise an error instead of guessing.

(imagetool-manager-organize)=

## Navigating and organizing tools

The left pane lists ImageTool windows, tools opened from ImageTool, and ImageTool
windows opened from those tools. Top-level ImageTool windows use an index and optional
data name (`index: name`). Rows derived from another row appear as child rows under the
row that made them. Selecting entries populates the right pane with details, a steps
list, and a preview of the main image.

:::{note}

Enable {menuselection}`View --> Preview on Hover` to see thumbnails while moving the mouse over
the list.

:::

Analysis tools and ImageTool windows opened from an ImageTool appear as child rows of
the ImageTool that opened them.

The following lists common actions included in the {guilabel}`File`, {guilabel}`Edit`,
and right-click context menus:

- {guilabel}`Show` / {guilabel}`Hide` / {guilabel}`Remove`

  Use the toolbar buttons or {kbd}`Return`, {kbd}`Ctrl+W`, and {kbd}`Del` to bring
  windows to the front, hide them, or remove them entirely. These controls live in
  {guilabel}`File`.

  :::{note}

  {kbd}`Ctrl+W` and {kbd}`Del` also works when analysis windows are focused, which is
  often more convenient than switching back to the manager.

  :::

- {guilabel}`Rename` / {guilabel}`Duplicate`

  Rename multiple selections at once or activate in-place editing for a single tool.
  {guilabel}`Duplicate` clones the currently selected windows, including their state.

- {guilabel}`Reset Index`

  Renumbers all windows from zero.

- {guilabel}`Link` / {guilabel}`Unlink`

  {kbd}`Ctrl+L` links the selected windows so they share cursors and slices;
  {kbd}`Ctrl+Shift+L` removes the links.

- {guilabel}`Offload to Workspace`

  Reloads the data as dask-backed data from the workspace file, freeing up memory but
  slowing down indexing. Use {menuselection}`Dask --> Load Into Memory` in ImageTool to load it
  back into memory when needed.

- {guilabel}`Concatenate`

  Combine selected data with {func}`xarray.concat` and open the result in a new
  ImageTool window.

- {guilabel}`Add to Figure…`

  Create a new {ref}`Figure Composer <figure-composer>` figure from the selected rows.
  When figures already exist, this action can also append a plotting step, add data
  sources without changing the recipe, or replace a source in the selected figure.

- {guilabel}`Reload Data`

  Recomputes selected data from its recorded source. For file-backed ImageTools, this
  re-fetches data from disk and reapplies any recorded operations. This is useful when
  conducting experiments, where you can repeat analysis on a continually updated file
  source with a single click.

  If the selected data cannot currently be reloaded, the action remains available and
  explains what is missing. Common fixes are reconnecting the drive that contains the
  source file, restoring a moved or deleted file, or reopening the ImageTool inputs that
  created the result.

  :::{note}

  This action is also available inside each ImageTool window, and is associated with the
  keyboard shortcut {kbd}`Ctrl+R` inside ImageTool windows.

  :::

- {guilabel}`Edit Note` / {guilabel}`Copy Note`

  Add plain-text notes to the selected row from the right-side inspector or the
  right-click context menu. Notes are saved with the manager workspace for ImageTool
  rows, analysis tools, child ImageTool outputs, and Figure Composer windows.

Icons next to each entry indicate special states: linked windows share a colored badge,
chunked Dask arrays show the dask icon, watched variables display their variable name,
rows opened from another row can show the state badges described in
{ref}`imagetool-manager-refresh`, and results that depend on several ImageTools can show
the {guilabel}`Changed` or {guilabel}`Missing` badges described in
{ref}`imagetool-manager-derived-data`.

(imagetool-manager-figure-composer)=

## Creating Matplotlib figures

Use {guilabel}`Add to Figure…` from the right-click context menu of one or more
ImageTool rows to send their data to {ref}`Figure Composer <figure-composer>`. When no
figures exist, the manager creates a new figure immediately. When figures already exist,
choose whether to create another figure, append a new recipe step, add the selected data
as sources only, or replace one source in the selected figure.

You can also drag ImageTool rows from the manager tree into an open Figure Composer
window to add them as sources.

New figures appear in a {guilabel}`Figures` tab in the manager.

See {ref}`Figure Composer <figure-composer>` for details.

(imagetool-manager-workspace)=

## Saving and loading

Windows in an ImageTool Manager instance can be saved to a workspace file (`.itws`),
similar to Igor Pro experiment files. Pressing {kbd}`Ctrl+S` in any child window saves
the entire manager workspace, including all windows and their state.

Manager row notes are workspace metadata. They are saved with the row they describe and
do not modify the data attributes of the underlying `DataArray`.

{menuselection}`File --> Add Windows From Workspace…` lets you choose windows from another
workspace file to import into the current one.

Saved ImageTool workspaces can be reloaded via {menuselection}`File --> Open Workspace…`
({kbd}`Ctrl+O`) or by dragging the `.itws` file back into the manager to recreate your
windows exactly as they were. A list of recent workspaces is available in
{menuselection}`File --> Open Recent`.

To check where the open manager is saved, choose {menuselection}`File --> Workspace Properties`
({kbd}`Alt+Return`).

Use {menuselection}`File --> Offload to Workspace` to make the selected data lazy-loaded from
the workspace file. This frees up memory but will slow down indexing and slicing. Use
{menuselection}`Dask --> Load Into Memory` in ImageTool to bring it back into memory.

If the workspace contains watched notebook variables, the watched rows reopen with
their variable-name badges. The rows stay disconnected until a notebook defines the
matching variables and reconnects them, as described in
{ref}`imagetool-manager-reconnect-watches`.

(imagetool-manager-nested-results)=

## Nested windows

When you are working in the manager, a new ImageTool window can appear as a child row
under the tool or ImageTool that created it. A typical session looks like this:

1. Open data in the manager with the methods described in {ref}`imagetool-manager-open`.
2. Launch {guilabel}`dtool`, {guilabel}`ktool`, or any other tool from that ImageTool.
3. Click {guilabel}`Open in ImageTool` in that tool.
4. The new ImageTool window appears under the tool or ImageTool that made it instead
   of as an unrelated top-level window.

That new ImageTool row remembers all of the information required to reproduce itself
from the raw data. When its parent node updates, the manager can automatically mark it
as out of date, update it, and show the steps in the side panel.

(imagetool-manager-result-placement)=

## Choosing where new data opens

Many ImageTool transform dialogs accessible from menu actions use
{guilabel}`Result Placement` to decide what happens to the transformed data:

- {guilabel}`Open Child Window` creates a new ImageTool row as a child of the current ImageTool.
- {guilabel}`Open Top-Level Window` creates a separate top-level ImageTool.
- {guilabel}`Replace Current` overwrites the active ImageTool with the transformed data.

(imagetool-manager-refresh)=

## Automatic updates

When data changes in an ImageTool or tool, the tools and ImageTool windows it created
may no longer match it. The manager shows this with badges:

- {guilabel}`Stale` means the ImageTool or tool that created this row changed, and this
  row can probably be updated.
- {guilabel}`Unavailable` means the manager cannot repeat the saved selection or
  operation on the current data, such as when a dimension, coordinate, or selection has
  changed too much.
- {guilabel}`Auto` means the row is up to date and automatic updates are enabled.

Click the badge in the tree or the update banner inside the tool window to review the
update. The {guilabel}`Automatic Updates…` dialog lets you apply a one-time update with
{guilabel}`Update Now`, or turn automatic updates on or off and save that preference
with {guilabel}`Save`. Saving only changes the automatic-update preference; it does not
refresh the current window immediately.

Fitting tools can also take part in this flow. {guilabel}`ftool`, {guilabel}`goldtool`,
and {guilabel}`restool` include {guilabel}`Refit after update`; when it is enabled, the
tool reruns the same fit after compatible data from the ImageTool that opened it is
updated.

(imagetool-manager-derived-data)=

## Results from several ImageTools

Some manager results are made from two or more top-level ImageTools instead of from one
parent row. Examples include {guilabel}`Concatenate` and console expressions such as
`tools[0] - tools[1]`.

The manager records the ImageTools that contributed to the result and shows their live
relationship in the tree and side panel:

- {guilabel}`Changed` means every recorded live input is still open, but at
  least one no longer matches the data or provenance that made the result.
- {guilabel}`Missing` means at least one recorded live input was removed from
  the manager.

These badges describe the relationship to the currently open inputs. They do not mean
the displayed result is invalid, and they do not change the result data automatically.

{guilabel}`Reload Data` also works on these results. If the live inputs are open, the
manager uses their current data. If an input was removed but its recorded file source is
still available, the manager reloads that input from the file before recomputing the
result.

If those recorded files moved, edit the file load row in the operation history and use
{guilabel}`Also relink selected file loads` to update the moved inputs together.

For results from console scripts, {guilabel}`Reload Data` replays the recorded code in
the console if possible. Only reload derived results from workspaces you trust.

(imagetool-manager-replay-code)=

## Operation history

Selecting a row fills the side panel with details about that item. For an ImageTool
window created from another row, the panel can show:

- The file(s) or data required to create the window
- The steps used to create the selected ImageTool window
- Code that can be pasted into a notebook or script to repeat those steps

Right-click on the steps list to copy code that rebuilds the data shown in the selected
ImageTool window. You can also copy selected steps, select another ImageTool in the
manager, and choose {guilabel}`Paste` to apply those steps to that ImageTool's
current data.

For [watched variables](working-with-notebooks), copied code contains the watched
variable name. File-backed workflows also include a snippet that loads the data in the
copied code. Otherwise, you will be prompted to enter the name of the variable to use as
the source when you copy code.

## Data Explorer and Console

In addition to ImageTool windows and analysis tools opened from them, the manager can
also launch standalone apps that stay outside the tree view and workspace files.

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

Run standard Python, `%magic` commands, or inspect objects with `?` exactly as you would
in a notebook.

(working-with-notebooks)=

## Notebook integration

The manager keeps notebooks synchronized through the `%watch` magic and provides
functions for retrieving and storing data.

:::{tip}
If you are using VS Code (or other editor that supports VS Code extensions), the dedicated `erlab` extension ( [marketplace](https://marketplace.visualstudio.com/items?itemName=khan.erlab) | [open-vsx](https://open-vsx.org/extension/khan/erlab) ) adds convenient commands for working with the manager directly from notebooks. Search for `erlab` in the extensions panel of your editor to install it.
:::

(imagetool-manager-watching)=

### Watching notebook variables

Load the IPython extension and start watching variables:

```python
%load_ext erlab.interactive
%watch my_data
```

:::{note}
To load the extension automatically, add `erlab.interactive` to your [IPython configuration](https://ipython.readthedocs.io/en/stable/config/intro.html), or configure VS Code’s Jupyter extension with

```json
"jupyter.runStartupCommands": [
    "%load_ext erlab.interactive"
]
```

so `%watch` is always available.
:::

Watching creates a labeled ImageTool window. Any time `my_data` changes in the notebook, the manager updates the matching window. Editing inside ImageTool—rotation, symmetrization, cropping, or other {ref}`ImageTool operations <imagetool-editing>`—writes the results back to the notebook variable.

:::{note}
To keep comparisons fast, only small subsets of large arrays are compared to check whether data has been modified. If a change slips by, re-run `%watch my_data` to force a refresh.
:::

Controlling watches:

```python
%watch data1 data2 data3   # add multiple variables
%watch                     # list watched names
%watch --restore           # reconnect saved watched rows by variable name
%watch -d data1 data2      # stop watching specific variables
%watch -x data1            # stop watching and close the window
%watch -z                  # stop watching everything
%watch -xz                 # stop watching and close every watched window
```

You can also right-click a tool in the manager and choose {guilabel}`Stop Watching`.

If a variable is deleted or replaced with a non-`DataArray`, the manager automatically
breaks the link and keeps the window as a regular ImageTool.

(imagetool-manager-reconnect-watches)=

#### Reconnecting watched rows

A watched row stores the variable name shown on its badge. If a notebook kernel stops,
or if you close and reopen the manager workspace, the row stays in the manager but
cannot synchronize until a notebook reconnects it. Disconnected watched rows keep
their variable-name badge and show a disconnected tooltip in the manager.

To reconnect one variable after restarting a notebook kernel:

1. Run the notebook cells that create the `DataArray`.
2. Run `%watch my_data` again.

The manager reuses the existing watched row for `my_data` instead of creating a
duplicate. This also forces a refresh if the automatic change detector missed an
update.

To reconnect every watched row in the open manager workspace:

```python
%watch --restore
```

`%watch --restore` looks at the watched variable names saved in the open workspace and
reconnects the rows whose names exist in the current notebook namespace as
`xarray.DataArray` objects. Rows for variables that are missing, or variables that
currently hold a different kind of object, stay disconnected.

To share a linked notebook and workspace with someone else:

1. Save the manager workspace as a `.itws` file.
2. Send both the `.ipynb` notebook and the `.itws` workspace.
3. On the other computer, open the `.itws` file in ImageTool Manager.
4. Run the notebook cells that create the watched variables.
5. Run `%watch --restore`.

The notebook and workspace do not need to live in the same folder. The rows reconnect
by the variable names saved in the workspace, so the receiving notebook must define
matching `DataArray` variables such as `my_data`.

If several disconnected watched rows use the same variable name, `%watch my_data` and
`%watch --restore` skip that name instead of guessing. Remove the extra watched links
with {guilabel}`Stop Watching`, or use an editor integration that can reconnect a
specific manager row.

#### Outside IPython (e.g., marimo notebooks)

If `%watch` is not available, use the Python API directly:

```python
from erlab.interactive.imagetool.manager import watch

# Start watching a DataArray
watch("my_data")

# Stop watching one variable
watch("my_data", stop=True)

# Stop watching everything
watch(stop_all=True)

# Reconnect saved watched rows in the open manager workspace
watch(restore=True)
```

In non-IPython environments, watcher updates fall back to polling, which periodically checks for changes in the watched variables. You can adjust the frequency with `poll_interval_s` if needed:

```python
watch("my_data", poll_interval_s=0.5)
```

Alternately, you can force an immediate check for changes with {func}`maybe_push <erlab.interactive.imagetool.manager.maybe_push>` instead of waiting for the next poll.

Use {func}`shutdown <erlab.interactive.imagetool.manager.shutdown>` to stop threads cleanly.

:::{note}

{func}`watch <erlab.interactive.imagetool.manager.watch>` can infer a namespace automatically, but providing an explicit `namespace=` argument is safer when you call it indirectly (for example, from small utility functions, callbacks, or wrappers) where the caller scope may not be obvious. In those cases, pass the exact mapping you want to watch, like `namespace=globals()`.

:::

(imagetool-manager-fetch)=

### Accessing manager data programmatically

Use {func}`fetch <erlab.interactive.imagetool.manager.fetch>` inside a notebook or script to copy data out of the manager:

```python
from erlab.interactive.imagetool.manager import fetch

data = fetch(0)  # returns an xarray.DataArray copy
```

Because `fetch` returns a copy, you can safely modify it without touching the live window.

### Sharing data via `%store`

The [%store](https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html) magic can persist variables between notebook sessions. Select tools in the manager and run {menuselection}`File --> Store with IPython` (or the matching context-menu command) to push their `DataArray` objects into the `%store` database. Internally, this executes:

```python
my_data = tools[0].data
%store my_data
```

Later, in any notebook, retrieve the stored variable with `%store -r my_data` and continue analysis without reopening files.

### Editor integration

If you are using VS Code (or other editor that supports VS Code extensions), the dedicated `erlab` extension ( [marketplace](https://marketplace.visualstudio.com/items?itemName=khan.erlab) | [open-vsx](https://open-vsx.org/extension/khan/erlab) ) adds convenient features for working with the manager directly from notebooks. Search for `erlab` in the extensions panel of your editor to install it.

(imagetool-manager-automation)=

## Automation APIs

If you wish to integrate the manager into custom workflows, you can programmatically load data and control ImageTool windows in the manager. Use the public functions exported from {mod}`erlab.interactive.imagetool.manager`:

```python
from erlab.interactive.imagetool.manager import load_in_manager, replace_data, show_in_manager

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

Under the hood these functions communicate with the GUI via ZeroMQ. Manager discovery is stored in a user-scoped live registry, so normal routing is intended for Python processes running in the same user session as the manager. See the API docs for details.

(imagetool-manager-standalone)=

## Installing as a standalone application

Standalone bundles for Windows and macOS let you run the manager without managing a
Python environment. They add OS-level conveniences such as opening supported files by
double-clicking them (or, on macOS, dropping files onto the Dock icon). For macOS 26 and
later, the app also features a dynamic icon that matches the new design language.

Download the latest release from the [project’s releases page](https://github.com/kmnhan/erlabpy/releases), then follow the platform-specific steps below. For other platforms, or if you prefer full control, build from source via {ref}`build-from-source`.

### Windows

1. Download the latest Windows build `.zip` file from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it and double-click the included `.exe` installer, then follow the prompts.

### macOS

1. Download the latest `.zip` archive that matches your architecture from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it to obtain `ImageTool Manager.app`.

3. Move the app into `/Applications` (or any folder you prefer) and launch it like any other macOS application.

### Linux and source-built bundles

Official standalone release bundles are currently only provided for Windows and macOS.
Linux users can build from source (see {ref}`build-from-source`), and the resulting app
can be launched directly from the build folder. For a more integrated experience, you
can create a desktop entry or alias that points to the built executable. If you want
supported builds for Linux, please submit an issue to let us know!

### Updating the application

Updates can be checked and installed from within the application itself. Select {guilabel}`Check for Updates` in the menu bar under {guilabel}`Help` (Windows) or {guilabel}`ImageTool Manager` next to {fab}`apple` (macOS) and follow the prompts.

(build-from-source)=

### Build from source

If you want to build the standalone application from source due to platform compatibility or other reasons, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/kmnhan/erlabpy.git
   cd erlabpy
   ```

2. Install dependencies (requires `uv`):

   ```bash
   uv sync --all-extras --group pyinstaller --group pyqt6
   ```

3. Build the application:

   ```bash
   uv run pyinstaller manager.spec
   ```

4. The resulting app will be in `dist/ImageTool Manager`.

5. *(Optional, Windows only)* Install [Inno Setup](https://jrsoftware.org/isinfo.php) and add to your system PATH. Then run `iscc manager.iss` to create an installer file.
