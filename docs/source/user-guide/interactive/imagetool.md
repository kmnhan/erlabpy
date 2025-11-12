# ImageTool

```{image} ../../images/imagetool_light.png
:align: center
:alt: ImageTool window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../../images/imagetool_dark.png
:align: center
:alt: ImageTool window in dark mode
:class: only-dark
```

:::

Inspired by *Image Tool* for Igor Pro, developed by the Advanced Light Source at Lawrence Berkeley National Laboratory, {class}`ImageTool <erlab.interactive.imagetool.ImageTool>` is a simple yet powerful tool for interactive image exploration.

## Features

- Zoom and pan
- Real-time slicing and binning
- Multiple cursors
- Easy size adjustment
- Advanced colormap controls
- Support for dask-backed data
- Interactive editing: rotation, normalization, cropping, momentum conversion, and more

## Getting started

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

- In an interactive session (like Jupyter), use the IPython magic command `%itool`:

  If you are working in an interactive session such as Jupyter, you can use the `%itool` magic command to display data in ImageTool. First, load the IPython extension:

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

  Then, display your data:

  ```python
  %itool data
  ```

  This command is equivalent to calling {func}`itool() <erlab.interactive.imagetool.itool>`. Many arguments to {func}`itool() <erlab.interactive.imagetool.itool>` are also available as options. For example, to open the data in the ImageTool manager, use the `--manager` option (or `-m` for short):

  ```python
  %itool -m data
  ```

  To see all supported arguments, run `%itool?` in an IPython session.

- If you use the ImageTool manager, you have more options for opening ImageTool windows. See [](imagetool-manager) for details.

## Tips

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

## Keyboard shortcuts

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
