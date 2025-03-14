# ImageTool

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

Inspired by *Image Tool* for Igor Pro written by the Advanced Light Source at Lawrence
Berkeley National Laboratory, {class}`ImageTool <erlab.interactive.imagetool.ImageTool>`
is a simple tool exploring images interactively.

### Features

- Zoom and pan
- Real-time slicing & binning
- Multiple cursors
- Easy size adjustment
- Advanced colormap control
- Interactive editing like rotation, normalization, cropping, momentum conversion, and
  more

### Displaying data in ImageTool

ImageTool supports *image-like* {class}`xarray.DataArray`s from 2 to 4 dimensions.
Non-uniform coordinates are converted to index arrays automatically.

Invoke ImageTool by calling {func}`itool <erlab.interactive.imagetool.itool>`:

```python
import erlab.interactive as eri
eri.itool(data)
```

Or use the {meth}`xarray.DataArray.qshow` accessor:

```python
data.qshow()
```

### Tips

- Hover over buttons for tooltips.

- Most actions have associated keyboard shortcuts. Explore the menu bar to learn them.

- Right-click on plots for context menus with options like copying slicing code, locking
  aspect ratio, exporting to a file, and more.

- Cursor controls

  - {material-regular}`grid_on`: snap cursors to pixel centers.
  - {material-regular}`add` and {material-regular}`remove`: add and remove
    cursors.

- Color controls

  - {material-regular}`brightness_auto`: lock color range and display a colorbar.

    - When toggled on, the color range is locked to the minimum and maximum of the entire
      data.
    - The color range can be manually set by dragging or right-clicking on the colorbar.

  - {material-regular}`vertical_align_center`: scale the
    colormap gamma with respect to the midpoint of the colormap, similar to
    {class}`CenteredPowerNorm <erlab.plotting.colors.CenteredPowerNorm>`.

  - {material-regular}`exposure`: switch between scaling behaviors of
    {class}`PowerNorm <matplotlib.colors.PowerNorm>` and {class}`InversePowerNorm
    <erlab.plotting.colors.InversePowerNorm>`.

  - You can choose different colormaps from the colormap dropdown menu. Only a subset of
    colormaps is loaded by default. To load all available colormaps, right-click on the
    colormap dropdown menu and select "Load All Colormaps".

- Binning controls

  - {material-regular}`settings_backup_restore`: reset all bin widths to 1.
  - {material-regular}`sync`: Apply binning changes to all cursors.

- Rotate and normalize data via the edit and view menus.

- ImageTool is extensible. At our home lab, we use a modified version of ImageTool to
  plot data as it is being collected in real-time!

### Keyboard shortcuts

Some shortcuts that are not shown in the menu bar. Mac users must replace {kbd}`Ctrl`
with {kbd}`⌘` and {kbd}`Alt` with {kbd}`⌥`.

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

Rule of thumb: hold {kbd}`Alt` to apply actions to all cursors. Shortcuts for 'shifting'
a cursor involves the {kbd}`Shift` key.

(imagetool-manager-guide)=

## Using the ImageTool manager

ImageTools can also be used as a standalone application with {class}`ImageToolManager
<erlab.interactive.imagetool.manager.ImageToolManager>`.

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

Run `itool-manager` in a terminal or command prompt window with ERLabPy installed.

:::{note}

- Only one manager can run per machine.
- Sending data to the manager has slight overhead, noticeable for large data.
:::

### Adding ImageTool windows

When the manager is running, new data can be opened in the manager by:

- Invoking ImageTool from {func}`itool <erlab.interactive.imagetool.itool>` or
  {meth}`xarray.DataArray.qshow` with `manager=True` from any script or notebook.

  ```python
  darr.qshow(manager=True)
  ```

- The `Move to Manager` ({kbd}`Ctrl+Shift+M`) action in the `File` menu from an
  ImageTool window opened without specifying `manager=True`. This action moves the
  active ImageTool to the manager.

- Opening supported files through the `File` menu in the manager.

- Dragging and dropping supported ARPES data into the manager window.

  In the dialog that appears, you can choose the plugin to use for loading the data.

  :::{hint}
  For scans that are recorded across multiple files, drag and dropping any file in the scan will automatically load and concatenate the entire scan. If you want to load only the file you dropped, choose the plugin suffixed with "Single File" in the dialog.
  :::

### Additional features

- Save all ImageTool windows to a file via the `Save Workspace As...` menu item.

  The saved windows can be restored later with `Open Workspace...` or by dragging and
  dropping the file into the manager.

- The manager has a built-in iPython console to manipulate ImageTool windows and data,
  and run Python code.

  Toggle the console with {kbd}`` ⌃+` `` (Mac) or {kbd}`` Ctrl+` `` (Windows/Linux) or through
  the `View` menu.

- Toggle the `Preview on Hover` option in the `View` menu to show a preview of the
  main image when hovering over each tool.

- After selecting multiple tools, you can perform actions on all selected tools at once
  using the right-click context menu.

- Selecting `Concatenate` from the right-click context menu will concatenate the data
  from all selected tools and open a new ImageTool window with the concatenated data.
  See {func}`xarray.concat <xarray.concat>` for more information on concatenation.

- The manager has an integrated file browser to browse and preview data files. It can be
  invoked from the `File` menu of the manager, or with the keyboard shortcut
  {kbd}`Ctrl+E`.

  See {mod}`erlab.interactive.explorer` for more information.

- Explore the menu bar for more features!

### Working with notebooks

Opening data in the manager from a notebook is straightforward, as shown above. However,
you may wish to send data from the manager to a jupyter notebook for further analysis.

This is easily done using the [%store](https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html) magic
command.

Suppose you want to store the data displayed in a tool with index 0. First select the
tool in the manager. Then, trigger the `Store with IPython` action from the
right-click context menu or the `File` menu. This will open a dialog to enter a
variable name. Enter a variable name (e.g., `my_data`) and click OK.

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
