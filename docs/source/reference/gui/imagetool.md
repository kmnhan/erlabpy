# ImageTool

Inspired by *Image Tool* for Igor Pro, developed by the Advanced Light Source at Lawrence Berkeley National Laboratory, {class}`ImageTool <erlab.interactive.imagetool.ImageTool>` delivers the same efficient workflow, now enhanced by {mod}`xarray` and Python.

(imagetool-features)=

## Key capabilities

- Responsive slicing of multidimensional (up to 4D) {class}`DataArray
  <xarray.DataArray>` objects, including Dask-backed data.
- Unlimited number of cursors with independent binning and code export for each line
  cut.
- Rich colormap controls with power law scaling, symmetric scaling about a center, and
  live color range adjustment.
- Built-in menus for selection, rotation, symmetrization, averaging, interpolation,
  cropping, coordinate reassignment, Fermi edge correction, and other common operations.
- Tight integration with tools such as {ref}`ktool <guide-ktool>`, {ref}`dtool
  <guide-dtool>`, and other tools listed in {ref}`interactive-misc-tools`, all
  accessible from ImageTool’s menus and context menus.
- Seamless integration with {ref}`ImageTool manager <imagetool-manager>` when you need
  to organize top-level ImageTool rows, tools opened from those ImageTools, ImageTool
  windows made by those tools, shared workspaces, or synchronized Jupyter notebooks.

For launch and notebook integration procedures, see
{ref}`how-to-gui-open-data-in-imagetool`.

(imagetool-launch)=
(imagetool-entry-points)=

## Opening ImageTool

ImageTool accepts these Python entry points:

- {meth}`xarray.DataArray.qshow` opens a DataArray:

  ```python
  data.qshow(link=True)
  ```

- {func}`erlab.interactive.imagetool.itool` opens one object or a list of objects:

  ```python
  import erlab.interactive as eri

  eri.itool(data, cmap="cividis")
  ```

  A list creates multiple windows. Set `link=True` to synchronize their cursor
  positions and bins. A Dataset or DataTree with multiple valid variables opens a
  variable-selection dialog.

- The `%itool` IPython magic opens a variable from the current namespace:

  ```python
  %load_ext erlab.interactive
  %itool data
  ```

  The `-m` or `--manager` option sends the new window to ImageTool Manager. Use
  `%itool --help` or `%itool?` for the complete option list.

- The [ERLab extension for VS Code](https://marketplace.visualstudio.com/items?itemName=khan.erlab)
  also opens DataArrays in ImageTool. The extension is also available from
  [Open VSX](https://open-vsx.org/extension/khan/erlab).

ImageTool adds a display axis for one-dimensional input. Singleton dimensions do not
count toward the four-dimensional limit. Input with more than four effective
dimensions opens the {guilabel}`Reduce Dimensions to Open` dialog. Each dimension can
be kept, reduced to one value, or aggregated. The dialog shows the resulting
dimensions and generated Python code before it opens the data.

(imagetool-interface)=

## Interface tour

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

Every ImageTool window is built from an {class}`ImageSlicerArea <erlab.interactive.imagetool.viewer.ImageSlicerArea>` plus dockable control panels:

- **Main image and cross-sections** – The central plot renders the current 2D slice. Orthogonal slices and cursor readouts update in real time as you move the cursors.

- **Cursor panel** – Add, remove, and modify cursors here. The coordinates of the active cursor are shown in editable text boxes.

- **Color panel** – Manipulate colormap normalization and appearance.

- **Binning panel** – Set bin widths per dimension and reset them with {material-regular}`settings_backup_restore`. Changes to the bin widths you make while {material-regular}`sync` is toggled are applied to all cursors.

(imagetool-data)=

## Working with dimensions and coordinates

ImageTool accepts image-like data with two to four effective dimensions. An effective
dimension has more than one value. One-dimensional input gains a display axis, and
singleton dimensions do not count toward the limit. Supported inputs include NumPy
arrays, {class}`xarray.DataArray`, {class}`xarray.Dataset`, and
{class}`xarray.DataTree`. Inputs with more than four effective dimensions open a
reduction dialog before ImageTool creates a window.

- The order of dimensions can be swapped using the arrow buttons in the cursor panel.
  The arrow points to the slice that will swap with the main view.

- Non-uniform coordinates are converted with a `_idx` suffix for plotting. Their true values are displayed in the cursor readouts.

- Use {menuselection}`Edit --> Edit Coordinates` to open the {guilabel}`Coordinate
  Editor` dialog. This dialog is an interface for
  {meth}`xarray.DataArray.assign_coords`. Use the dialog to do these actions:

  - Set the start and end values.
  - Edit individual values.
  - Scale and offset a numeric scalar or 1D coordinate with
    `new = scale * old + offset`.
  - Add a scalar coordinate.
  - Add a 1D associated coordinate along an existing dimension.

- Use {menuselection}`Edit --> Edit Attributes` to open the {guilabel}`Attribute Editor`
  dialog. This is a GUI for {meth}`xarray.DataArray.assign_attrs` that lets you change
  existing attributes or add new typed attributes while leaving untouched attributes in
  place. Choose {guilabel}`String`, {guilabel}`Int`, {guilabel}`Float`,
  {guilabel}`Bool`, or {guilabel}`Python literal` when entering values.

- Use {menuselection}`Edit --> Rename…` to open the {guilabel}`Rename Coordinates and
  Dimensions` dialog. This is a GUI for {meth}`xarray.DataArray.rename` that lets you
  rename coordinates and dimensions.

- Use {menuselection}`Edit --> Swap Dimensions` to open the {guilabel}`Swap Dimensions` dialog.
  This is an interface for {meth}`xarray.DataArray.swap_dims`.

- Dask-backed arrays are fully supported. The dedicated {guilabel}`Dask` menu exposes
  actions to compute the array into memory, rechunk automatically, or choose custom
  chunk shapes within ImageTool.

- Overlay plots of numeric non-dimensional coordinates, such as temperature, on profile
  plots from {menuselection}`View --> Plot Associated Coordinates`. Multi-dimensional
  coordinates are sliced with the active cursor and averaged over binned hidden
  dimensions. Right-click a profile plot to open associated coordinates in a new
  ImageTool window.

- Use {menuselection}`View --> Set Cursor Colors by Coordinate…` to color cursors by a
  dimension coordinate or numeric associated coordinate value at each cursor position.

(imagetool-slicing)=

## Slicing and binning

- Drag with the left mouse button to pan. Drag with the right mouse button or use the
  wheel to zoom. Scroll on an individual axis to zoom only that dimension.

- Drag a cursor line to change the slicing position. You can also hold {kbd}`Ctrl` and
  drag on a plot. The {menuselection}`View --> Cursor Control` submenu lists keyboard
  commands for moving the active cursor.

- Binning displays the average over the selected bin width. Shaded regions beside the
  cursor lines show the averaged range. Binning does not change the stored data.

- {menuselection}`Edit --> Aggregate…` and {guilabel}`Coarsen` reduce the underlying
  data. Use these operations when the result must contain fewer data points.

For a procedure that synchronizes slices and bins across windows, see
{ref}`how-to-gui-compare-linked-data`.

(imagetool-cursors)=

## Cursor control and context menus

- Hover over any toolbar icon to see a short description of its function.

- Copy the numeric readouts at any time with {kbd}`Ctrl+Shift+C` (cursor values) or
  {kbd}`Ctrl+Alt+C` (cursor indices). ImageTool copies native Python literals so you can
  paste them directly into scripts.

- Right-click the data value readout in the cursor panel to switch it between the data
  value and any currently plotted associated coordinate value at the active cursor.

- Multiple cursors can be added to the image using the {material-regular}`add` button in
  the cursor panel. They can each be dragged independently, and their bin widths can be
  set separately in the binning panel. To switch the active cursor, simply click on it
  or select it from dropdown menu in the cursor panel.

- To move all cursors simultaneously, hold {kbd}`Alt` while dragging a cursor line, or
  use {kbd}`Ctrl+Alt` while dragging on the image.

- Right-click on an image plot or line plot to open a useful context menu. Common
  options include copying the slicing code, locking the aspect ratio on image plots,
  exporting the current selection, and opening tools.

  On image plots, the context menu can launch {ref}`goldtool <guide-goldtool>`,
  {ref}`restool <guide-restool>`, {ref}`dtool <guide-dtool>`, and {ref}`ftool
  <guide-ftool>`. On line plots, the context menu offers {ref}`ftool <guide-ftool>`.
  Image and line plot context menus can also send the current plot to
  {ref}`Figure Composer <figure-composer>`.

  Tools opened from ImageTool remember the slice or selection that opened them. If that
  ImageTool is updated with compatible data, the tool shows a {guilabel}`Stale` badge
  instead of silently keeping old input. Click the badge inside the tool window to
  update it from the latest compatible data, or enable automatic updates for future
  changes. When ImageTool is open in the manager, ImageTool windows opened from those
  tools can appear as child rows under the tool row; see
  {ref}`imagetool-manager-nested-results` and {ref}`imagetool-manager-refresh`.

  :::{hint}
  Holding {kbd}`Alt` while opening the menu switches many actions to cropped mode, which crops the data to what is currently visible in the plot before performing the action. This is useful for conducting analysis on a specific region.
  :::

- Use {menuselection}`View --> Rotation Guidelines` to add guidelines for azimuthal offsets or
  symmetry operations.

  The guideline center moves together with the cursor. The center and the angle of the
  guidelines feed directly into the {guilabel}`Rotate` dialog and {guilabel}`ktool` for
  fast alignment.

- Use {menuselection}`View --> Open ktool` and {menuselection}`View --> Open meshtool` for tools
  launched from the main menu rather than the plot context menu.

- The default color cycle of cursors is user configurable. See
  {doc}`ImageTool settings <settings>`.

- Colors can be changed individually from {menuselection}`View --> Edit Cursor Colors…`, where
  you can choose from a colormap or edit each cursor's color separately.

- Alternatively, the colors of the cursors can be set to follow a specific coordinate
  dynamically based on their positions. This can be enabled from
  {menuselection}`View --> Set Cursor Colors by Coordinate…`.

(imagetool-editing)=

## Data operations

Editing dialogs are available from the {guilabel}`Edit` and {guilabel}`View` menus.
Most editing dialogs can replace the current data or open the result separately. When
ImageTool is managed, {guilabel}`Result Placement` can create a child row, create a
top-level row, or replace the current row.

- {menuselection}`Edit --> Rotate` applies
  {func}`erlab.analysis.transform.rotate`. A visible rotation guideline supplies the
  initial angle and center.
- {menuselection}`Edit --> Select Data…` builds selections with
  {meth}`xarray.DataArray.qsel`, {meth}`xarray.DataArray.sel`, or
  {meth}`xarray.DataArray.isel`.
- {menuselection}`Edit --> Aggregate…` reduces selected dimensions with mean, minimum,
  maximum, or sum.
- {menuselection}`Edit --> Interpolate…` calls {meth}`xarray.DataArray.interp` along one
  dimension with `linear` or `nearest` interpolation.
- {menuselection}`Edit --> Sort By…` calls {meth}`xarray.DataArray.sortby` with one or
  more coordinate keys.
- {menuselection}`Edit --> Leading Edge…` calls
  {func}`erlab.analysis.interpolate.leading_edge` along a selected dimension.
- {menuselection}`Edit --> Coarsen` provides window, boundary, side, coordinate, and
  reduction controls for {meth}`xarray.DataArray.coarsen`.
- {menuselection}`Edit --> Thin` calls {meth}`xarray.DataArray.thin`.
- {menuselection}`Edit --> Symmetrize --> Mirror…` applies reflection or
  antisymmetrization about a selected coordinate.
- {menuselection}`Edit --> Symmetrize --> Rotational…` applies rotational
  symmetrization. A visible rotation guideline supplies the initial center and fold
  count.
- {menuselection}`Edit --> Crop` selects data between cursors.
  {menuselection}`Edit --> Crop to View` selects the visible coordinate range.
- {menuselection}`Edit --> Correct With Edge…` loads a fitted edge with
  {func}`xarray_lmfit.load_fit` and shifts data along `eV`.
- {menuselection}`View --> Normalize` applies a reversible display filter with area,
  min-max, and baseline options.
- {menuselection}`View --> Gaussian Filter` applies a reversible, coordinate-aware
  Gaussian display filter.

{menuselection}`Edit --> Undo` and {menuselection}`Edit --> Redo` move through editing
history. {menuselection}`View --> Reset` removes the active display filter. Opening the
active filter dialog again starts from and replaces its current settings.

(imagetool-color)=

## Color and normalization

- Toggle {material-regular}`brightness_auto` to lock the color range to the global data
  min/max and display a colorbar alongside the image. Drag on the colorbar to update
  limits interactively or right-click to type exact bounds.

- {material-regular}`vertical_align_center` applies gamma scaling relative to the
  midpoint, which is handy for centered intensity scales such as spin-polarized or
  dichroic data.

- Use {material-regular}`exposure` to flip between normalization behaviors of
  {class}`matplotlib.colors.PowerNorm` and
  {class}`erlab.plotting.colors.InversePowerNorm`.

- By default, only a subset of Matplotlib colormaps is loaded. You can load the whole
  catalog by right-clicking on the colormap drop-down and selecting {guilabel}`Load All
  Colormaps`.

For ROI procedures, see {ref}`how-to-gui-extract-polygon-path` and
{ref}`how-to-gui-mask-polygon`.

For export and settings procedures, see {ref}`imagetool-export`.

(imagetool-shortcuts)=

## Keyboard shortcuts

Most actions advertise their shortcut directly in the menu bar. The table below
highlights common gestures. Replace {kbd}`Ctrl` with {kbd}`⌘` and {kbd}`Alt` with
{kbd}`⌥` on macOS.

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
  - Move all cursors simultaneously
- - {kbd}`Alt` while dragging a cursor line
  - Move all cursor lines along
:::

Rule of thumb: hold {kbd}`Alt` to apply actions to all cursors. Shortcuts for 'shifting'
a cursor involves the {kbd}`Shift` key.
