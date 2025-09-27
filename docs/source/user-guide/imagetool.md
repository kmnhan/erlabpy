# ImageTool

The interactive image viewer and analyzer for ARPES spectra and other image-like data.

Workflow mainly involves a single ImageTool window for quick interactive exploration, with optional use of the ImageTool manager to organize and manage multiple ImageTool windows.

## The ImageTool window

```{image} ../images/imagetool_light.png
:align: center
:alt: ImageTool window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../images/imagetool_dark.png
:align: center
:alt: ImageTool window in dark mode
:class: only-dark
```

:::

Inspired by *Image Tool* for Igor Pro, developed by the Advanced Light Source at Lawrence Berkeley National Laboratory, {class}`ImageTool <erlab.interactive.imagetool.ImageTool>` is a simple yet powerful tool for interactive image exploration.

### Features

- Zoom and pan
- Real-time slicing and binning
- Multiple cursors
- Easy size adjustment
- Advanced colormap controls
- Interactive editing: rotation, normalization, cropping, momentum conversion, and more

### Displaying data in ImageTool

ImageTool supports *image-like* {class}`xarray.DataArray` objects with 2 to 4 dimensions. Non-uniform coordinates are automatically converted to index arrays, which are suffixed with `_idx`.

There are several ways to display data in ImageTool:

- Using {func}`itool() <erlab.interactive.imagetool.itool>`:

  ```python
  import erlab.interactive as eri
  eri.itool(data)
  ```

- Using the {meth}`xarray.DataArray.qshow` accessor:

  ```python
  data.qshow()
  ```

- In an interactive session, use the IPython magic command `%itool`:

  If you are working in IPython or a Jupyter notebook, you can use the `%itool` magic command to display data in ImageTool. First, load the IPython extension:

  ```python
  %load_ext erlab.interactive
  ```

  Then, display your data:

  ```python
  %itool data
  ```

  This command is equivalent to calling {func}`itool() <erlab.interactive.imagetool.itool>`. Many arguments to {func}`itool() <erlab.interactive.imagetool.itool>` are also available as options. For example, to open the data in the ImageTool manager, use the `--manager` option (or `-m` for short):

  ```python
  %itool -m data
  ```

  To see all supported arguments, run `%itool?` in an IPython session.

- Working with the ImageTool manager: see the [next section](#imagetool-manager).

### Tips

- Hover over buttons for tooltips.

- Most actions have associated keyboard shortcuts. Explore the menu bar to learn them.

- Right-click on plots for context menus with options like copying slicing code, locking aspect ratio, exporting to a file, opening various tools, and more.

  :::{hint}
  Holding {kbd}`Alt` inside the context menu will transform some menu items to work with the data cropped to the currently visible area.
  :::

- Cursor controls

  - {material-regular}`grid_on`: snap cursors to pixel centers.
  - {material-regular}`add` and {material-regular}`remove`: add and remove cursors.

- Color controls

  - {material-regular}`brightness_auto`: lock color range and display a colorbar.

    - When toggled on, the color range is locked to the minimum and maximum of the entire data.
    - The color range can be manually set by dragging or right-clicking on the colorbar.

  - {material-regular}`vertical_align_center`: scale the colormap gamma with respect to the midpoint of the colormap, similar to {class}`CenteredPowerNorm <erlab.plotting.colors.CenteredPowerNorm>`.

  - {material-regular}`exposure`: switch between scaling behaviors of {class}`PowerNorm <matplotlib.colors.PowerNorm>` and {class}`InversePowerNorm <erlab.plotting.colors.InversePowerNorm>`.

  - You can choose different colormaps from the colormap dropdown menu. Only a subset of colormaps is loaded by default. To load all available colormaps, right-click on the colormap dropdown menu and select "Load All Colormaps".

- Binning controls

  - {material-regular}`settings_backup_restore`: reset all bin widths to 1.
  - {material-regular}`sync`: Apply binning changes to all cursors.

- In the "Edit" and "View" menu bar items, you can find various options to edit and transform the data, such as rotating, symmetrizing, and cropping. Try them out!

- ImageTool is extensible. At our home lab, we use a modified version of ImageTool to plot data as it is being collected in real-time!

### Keyboard shortcuts

Some shortcuts that are not shown in the menu bar. Mac users must replace {kbd}`Ctrl` with {kbd}`⌘` and {kbd}`Alt` with {kbd}`⌥`.

:::{list-table}
:header-rows: 1

- - Shortcut
  - Description
- - {kbd}`LMB` Drag
  - Pan
- - {kbd}`RMB` Drag
  - Zoom and scale
- - {kbd}`Ctrl+LMB` Drag
  - Move active cursor
- - {kbd}`Ctrl+Alt+LMB` Drag
  - Move all cursors
- - {kbd}`Alt` while dragging a cursor line
  - Move all cursor lines
:::

Rule of thumb: hold {kbd}`Alt` to apply actions to all cursors. Shortcuts for 'shifting' a cursor involves the {kbd}`Shift` key.

(imagetool-manager)=

## ImageTool manager

ImageTools can also be used as a standalone application with {class}`ImageToolManager <erlab.interactive.imagetool.manager.ImageToolManager>`.

```{image} ../images/manager_light.png
:align: center
:alt: ImageToolManager window screenshot
:class: only-light
:width: 600px
```

:::{only} format_html

```{image} ../images/manager_dark.png
:align: center
:alt: ImageToolManager window screenshot
:class: only-dark
:width: 600px
```

:::

The manager shows a list of opened ImageTool windows along with some buttons. Information and preview about the data displayed in the currently selected ImageTool is shown in the right panel.

:::{hint}
Hovering your mouse over each button will show a brief description of its function.
:::

### Starting the manager

Run `itool-manager` in a terminal or command prompt window in an environment where ERLabPy is installed.

:::{note}

- Only one manager can run per machine.
- Sending data to the manager has slight overhead, noticeable for large data.
:::

### Adding ImageTool windows

When the manager is running, new data can be opened in the manager by:

- Invoking ImageTool from {func}`itool() <erlab.interactive.imagetool.itool>` or {meth}`xarray.DataArray.qshow` with `manager=True`.

  ```python
  darr.qshow(manager=True)
  ```

- Using the `%itool` magic command with the `--manager` (or `-m`) option in an IPython session or Jupyter notebook.

  ```python
  %itool -m darr
  ```

- Watching a variable in a Jupyter notebook with the `%watch` magic command.

  See [Working with notebooks](#working-with-notebooks) below for more information.

- The `Move to Manager` ({kbd}`Ctrl+Shift+M`) action in the `File` menu from an ImageTool window opened without specifying `manager=True`. This action moves the active ImageTool to the manager.

- Opening supported files through the `File` menu in the manager.

- Dragging and dropping supported ARPES data into the manager window.

  In the dialog that appears, you can choose the plugin to use for loading the data.

  :::{hint}
  For scans that are recorded across multiple files, drag and dropping any file in the scan will automatically load and concatenate the entire scan. If you want to load only the file you dropped, choose the plugin suffixed with "Single File" in the dialog.
  :::

### Saving and loading workspaces

You can save all open ImageTool windows to an HDF5 file using the `Save Workspace As...` menu item in the manager. Later, restore your workspace with `Open Workspace...` or by dragging and dropping the workspace file into the manager. Colormaps, cursor positions, window sizes, and other settings are preserved. Workspace files are portable and can be shared with others, who can open them in their own ImageTool manager.

### Additional features

- Replace data in an existing ImageTool window by supplying the `replace` argument to {func}`itool() <erlab.interactive.imagetool.itool>` or {meth}`xarray.DataArray.qshow`:

  ```python
  data.qshow(manager=True, replace=1)
  ```

  To replace data in multiple windows at once:

  ```python
  eri.itool([data1, data2], manager=True, replace=[1, 2])
  ```

- Save all ImageTool windows to a file via the `Save Workspace As...` menu item.

  The saved windows can be restored later with `Open Workspace...` or by dragging and dropping the file into the manager.

- The manager includes a built-in IPython console for manipulating ImageTool windows and data, and running Python code.

  Toggle the console with {kbd}`⌃+` (Mac) or {kbd}`Ctrl+` (Windows/Linux), or use the `View` menu.

  :::{hint}
  The console provides access to the `tools` list, which contains all open ImageTool windows. You can manipulate each window’s data via the `data` attribute, e.g., `tools[0].data`.
  The console is a full-featured IPython environment, supporting tab completion, magic commands, and other IPython features. For example, use the `?` operator to view function signatures and docstrings, e.g., `xr.concat?`.
  :::

- Enable the `Preview on Hover` option in the `View` menu to show a preview of the main image when hovering over each tool.

- After selecting multiple tools, you can perform actions on all selected tools at once using the right-click context menu.

- Use the `Concatenate` option in the right-click context menu to combine data from all selected tools and open a new ImageTool window with the concatenated data. See {func}`xarray.concat <xarray.concat>` for details.

- The manager features an integrated file browser for browsing and previewing data files. Access it from the `File` menu or with the keyboard shortcut {kbd}`Ctrl+E`.

  See {mod}`erlab.interactive.explorer` for more information.

- Explore the menu bar for additional features!

(working-with-notebooks)=

### Working with notebooks

#### Synchronization

The ImageTool manager can automatically synchronize a Jupyter notebook variable with an ImageTool window. When the variable changes, the window updates automatically. If you transform the data in ImageTool (e.g., rotate, symmetrize, crop), the notebook variable is updated as well. We call this feature *watching* a variable.

First, load the IPython extension in your notebook:

```python
%load_ext erlab.interactive
```

:::{note}
To load the extension automatically in new notebooks, add `erlab.interactive` to your IPython config. See the [IPython documentation](https://ipython.readthedocs.io/en/stable/config/intro.html) for details.

If you use VS Code with the Jupyter extension, add this to your workspace or user `settings.json`:

```json
"jupyter.runStartupCommands": [
    "%load_ext erlab.interactive"
]
```

:::

Then watch a variable (e.g., `my_data`):

```python
%watch my_data
```

This opens a new ImageTool window in the manager displaying `my_data`. When `my_data` changes, the window updates automatically. A label with the variable name appears next to the tool in the manager.

:::{note}
Change detection runs after each cell execution. To avoid slow comparisons, only a subset of large arrays is checked. If an update is missed, force a refresh by re-running `%watch my_data`.
:::

You can watch multiple variables:

```python
%watch data1 data2 data3
```

List watched variables:

```python
%watch
```

Stop watching specific variables:

```python
%watch -d data1 data2
```

You can also stop watching from the tool’s right-click context menu in the manager.

If a variable is deleted or changed to a non-compatible type, it stops being watched:

```python
del data1
data2 = "not a DataArray anymore"
```

The corresponding ImageTool windows become regular windows and no longer update automatically.

To stop watching and also close the corresponding windows:

```python
%watch -x data1 data2
```

Stop watching all variables:

```python
%watch -z
```

Combine with `-x` to also close all corresponding windows:

```python
%watch -xz
```

:::{note}
If you close the notebook or restart the kernel, watched variables remain in the manager but are no longer synchronized with the notebook. The corresponding windows keep their labels, but they no longer update automatically. You can remove the labels by right-clicking the tools and choosing "Stop Watching". To avoid confusion, stop watching all variables before closing the notebook or restarting the kernel.
:::

#### Accessing data from a notebook

Fetch data from a manager window by index with {func}`erlab.interactive.imagetool.manager.fetch`:

```python
from erlab.interactive.imagetool.manager import fetch
```

```python
data = fetch(0)
```

:::{note}
The fetched data is a copy. Modifying it does not affect the displayed data.
:::

#### Integration with `%store` magic command

The [%store](https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html) magic command in IPython allows you to persist variables across different sessions. You can use this feature to store data from an ImageTool window in the manager and retrieve it later in any notebook session. The manager has built-in support for this feature, making it easy to store data from an ImageTool window using the `%store` magic command.

Suppose you want to store the data displayed in a tool with index 0. First select the tool in the manager. Then, trigger the `Store with IPython` action from the right-click context menu or the `File` menu. This will open a dialog to enter a variable name. Enter a variable name (e.g., `my_data`) and click OK.

:::{note}
This is equivalent to running the following code in the manager console:

```python
my_data = tools[0].data
%store my_data
```

:::

Now, in any notebook, you can retrieve the data by running:

```python
%store -r my_data
```

after which the data will be available as `my_data` in the notebook.
