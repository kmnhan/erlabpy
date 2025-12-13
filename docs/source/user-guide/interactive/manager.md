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

{class}`ImageToolManager <erlab.interactive.imagetool.manager.ImageToolManager>` keeps large analysis sessions organized. It tracks every ImageTool window, preview, and metadata entry in a single tree view so you can link, archive, and share them without juggling dozens of floating windows.

(imagetool-manager-overview)=

## Why use the manager?

- Launch and watch many ImageTool windows simultaneously without interrupting your notebook or script.
- Link multiple ImageTools, duplicate them, or update their data in place in case of real-time data acquisition.
- Save and reopen complete workspaces, including colormaps, cursor positions, window geometry, and ROIs.
- Archive rarely used windows to disk so they can be restored later without consuming RAM.
- Synchronize directly with Jupyter via `%watch`, access data from scripts using {func}`fetch <erlab.interactive.imagetool.manager.fetch>`, and perform quick analyses through a built-in IPython console.
- Drag-and-drop files to open them quickly, or use the integrated data explorer to browse preview data.

(imagetool-manager-start)=

## Starting the manager

Run `itool-manager` in a terminal or command prompt window in an environment where ERLabPy is installed.

:::{note}

- Only one manager can run per machine.

- Opening an ImageTool window for the very first time after installing may take a couple of minutes as caches are built. Subsequent launches will be much faster.

- The manager can be installed as a packaged build which enables some convenient features as described in {ref}`imagetool-manager-standalone`.

:::

(imagetool-manager-open)=

## Opening and replacing ImageTool windows

Once the manager is running, you can open ImageTools in several ways:

- {meth}`xarray.DataArray.qshow` or {func}`erlab.interactive.imagetool.itool` with `manager=True` sends windows directly to the manager:

  ```python
  data.qshow(manager=True)
  eri.itool([d1, d2], manager=True, replace=[1, 2])
  ```

  Pass `replace=` to update data in existing windows instead of creating new ones.

- {ref}`ImageTool’s %itool magic command <imagetool-entry-points>` with the `--manager` (or `-m`) flag in an IPython session or Jupyter notebook.

  ```python
  %itool -m darr
  ```

- The {guilabel}`File → Move to Manager` ({kbd}`Ctrl+Shift+M`) action from an ImageTool window opened outside the manager. This action moves the active ImageTool to the manager.

- Use the manager’s {guilabel}`File → Open File…` action to load data in a new ImageTool.

- Drag and drop supported ARPES data into the manager window.

  In the dialog that appears, you can choose the plugin to use for loading the data.

  :::{hint}
  For scans that are recorded across multiple files, drag and dropping any file in the scan will automatically load and concatenate the entire scan. If you want to load only the file you dropped, choose the plugin suffixed with {guilabel}`Single File` in the dialog.
  :::

- Launch the built-in data explorer from {guilabel}`File → Data Explorer` or {kbd}`Ctrl+E`. Browse arbitrary folders, preview metadata, and open selected files in the manager.

- Watch notebook variables with the `%watch` magic to create windows that stay synchronized with your data structures. See {ref}`working-with-notebooks`.

- For custom integration with other workflows, scripts can call {func}`erlab.interactive.imagetool.manager.show_in_manager` or {func}`~erlab.interactive.imagetool.manager.load_in_manager` directly (see {ref}`imagetool-manager-automation`).

(imagetool-manager-organize)=

## Navigating and organizing tools

The left pane lists every ImageTool window by index and optional name (`index: name`). Selecting entries populates the right pane with metadata and a live preview.

:::{note}
Enable {guilabel}`View → Preview on Hover` to see thumbnails while moving the mouse over the list.
:::

The following lists common actions included in the {guilabel}`File`, {guilabel}`Edit`, and right-click context menus:

- {guilabel}`Show` / {guilabel}`Hide` / {guilabel}`Remove` – Use the toolbar buttons or {kbd}`Return`, {kbd}`Ctrl+W`, and {kbd}`Del` to bring windows to the front, hide them, or remove them entirely. These controls live in {guilabel}`File`.
  :::{note}
  {kbd}`Ctrl+W` and {kbd}`Del` also works when analysis windows are focused, which is often more convenient than switching back to the manager.
  :::
- {guilabel}`Rename` / {guilabel}`Duplicate` – Rename multiple selections at once or activate in-place editing for a single tool. {guilabel}`Duplicate` clones the currently selected windows, including their state.
- {guilabel}`Reset Index` – Renumbers all windows from zero.
- {guilabel}`Link` / {guilabel}`Unlink` – {kbd}`Ctrl+L` links the selected windows so they share cursors and slices; {kbd}`Ctrl+Shift+L` removes the links.
- {guilabel}`Archive` / {guilabel}`Unarchive` – {guilabel}`Archive` writes a tool’s dataset and UI state to a temporary file, frees its memory, and grays it out in the list. {guilabel}`Unarchive` reopens it. Both actions are under {guilabel}`File`.
- {guilabel}`Concatenate` – Combine selected data with {func}`xarray.concat` and open the result in a new ImageTool window.
- {guilabel}`Reload Data` – Re-fetches data from disk using the original loader function. Handy when data is updated during acquisition.

Icons next to each entry indicate special states: linked windows share a colored badge, chunked Dask arrays show the dask icon, and watched variables display their notebook name. Right-click to see all context-sensitive actions.

(imagetool-manager-archive-workspace)=

## Workspaces

Choose {guilabel}`File → Save Workspace As…` to save multiple open windows to a single `.itws` file. Workspaces store not only the data, but also the ImageTool settings such as cursor locations, colormaps, window geometry, and ROIs.

Saved ImageTool workspaces can be reloaded via {guilabel}`File → Open Workspace…` or by dragging the `.itws` file back into the manager to recreate your windows exactly as they were. Share the file with collaborators and they will see the identical layout.

## Built-in explorer and console

- {guilabel}`Data Explorer` – The explorer window ({guilabel}`File → Data Explorer` or {kbd}`Ctrl+E`) provides a filesystem browser tailored for ARPES datasets. Preview metadata, queue batch loads, or open entire directories as tabs.

- {guilabel}`Console` – Toggle the embedded IPython console with {kbd}`⌃+` on macOS or {kbd}`Ctrl+` on Windows/Linux, or via the {guilabel}`View` menu. The console exposes a `tools` list containing wrappers for every ImageTool. For example:

  ```python
  # List names of all windows
  [tool.name for tool in tools]

  # Access the underlying DataArray of the first window
  tools[0].data

  # Replace data in the first window
  tools[0].data = new_data
  ```

  Run standard Python, `%magic` commands, or inspect objects with `?` exactly as you would in a notebook.

(working-with-notebooks)=

## Notebook integration

The manager keeps notebooks synchronized through the `%watch` magic and exposes helper APIs for retrieving and storing data.

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
%watch -d data1 data2      # stop watching specific variables
%watch -x data1            # stop watching and close the window
%watch -z                  # stop watching everything
%watch -xz                 # stop watching and close every watched window
```

You can also right-click a tool in the manager and choose {guilabel}`Stop Watching`.

If a variable is deleted or replaced with a non-`DataArray`, the manager automatically breaks the link and keeps the window as a regular ImageTool.

:::{note}
When a notebook kernel shuts down, watched windows remain open in  but no longer synchronize. Use {guilabel}`Stop Watching` or run `%watch -z` before closing the kernel to avoid confusion. Variables watched from different notebooks are color-coded for clarity.
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

The [%store](https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html) magic can persist variables between notebook sessions. Select tools in the manager and run {guilabel}`File → Store with IPython` (or the matching context-menu command) to push their `DataArray` objects into the `%store` database. Internally, this executes:

```python
my_data = tools[0].data
%store my_data
```

Later, in any notebook, retrieve the stored variable with `%store -r my_data` and continue analysis without reopening files.

(imagetool-manager-automation)=

## Automation APIs

If you wish to integrate the manager into custom workflows, you can programmatically load data and control ImageTool windows in the manager. Use the public helpers exported from {mod}`erlab.interactive.imagetool.manager`:

```python
from erlab.interactive.imagetool.manager import load_in_manager, show_in_manager

# Open raw files and let the manager choose the loader interactively
load_in_manager(["scan1.pxt", "scan2.pxt"])

# Open two ImageTools and link their cursors
show_in_manager([data_a, data_b], link=True)

# Replace the dataset at index 3 with a new result
show_in_manager(new_data, replace=3)
```

Additional functions such as {func}`replace_data <erlab.interactive.imagetool.manager.replace_data>`, {func}`watch_data <erlab.interactive.imagetool.manager.watch_data>`, and {func}`unwatch_data <erlab.interactive.imagetool.manager.unwatch_data>` give you finer control when building custom acquisition pipelines.

Under the hood these helpers communicate with the GUI via ZeroMQ, so they can be called from any Python process that can reach the manager (even on a different machine). See the API docs for details.

(imagetool-manager-standalone)=

## Installing as a standalone application

Standalone bundles for Windows and macOS let you run the manager without managing a Python environment. They add OS-level conveniences such as opening supported files by double-clicking them (or, on macOS, dropping files onto the Dock icon). For macOS 26 and later, the app also features a dynamic icon that matches the new design language.

Standalone bundles for Windows and macOS let you run the manager without managing a Python environment. They add OS-level conveniences such as opening supported files by double-clicking them (or, on macOS, dropping files onto the Dock icon) and include a macOS “liquid glass” icon for clarity.

Download the latest release from the [project’s releases page](https://github.com/kmnhan/erlabpy/releases), then follow the platform-specific steps below. For other platforms, or if you prefer full control, build from source via {ref}`build-from-source`.

### Windows

1. Download the latest Windows build `.zip` file from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it and double-click the included `.exe` installer, then follow the prompts.

### macOS

1. Download the latest `.zip` archive that matches your architecture from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it to obtain `ImageTool Manager.app`.

3. Move the app into `/Applications` (or any folder you prefer) and launch it like any other macOS application.

### Updating the application

Updates can be checked and installed from within the application itself. Select {guilabel}`Check for Updates...` in the menu bar under {guilabel}`Help` (Windows) or {guilabel}`ImageTool Manager` next to {fab}`apple` (macOS) and follow the prompts.

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
