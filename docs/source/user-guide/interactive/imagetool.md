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

Inspired by *Image Tool* for Igor Pro, developed by the Advanced Light Source at Lawrence Berkeley National Laboratory, {class}`ImageTool <erlab.interactive.imagetool.ImageTool>` delivers the same efficient workflow, now enhanced by {mod}`xarray` and Python.

(imagetool-features)=

## Key capabilities

- Responsive slicing of multidimensional (up to 4D) {class}`DataArray <xarray.DataArray>` objects, including Dask-backed data.
- Unlimited number of cursors with independent binning and code export for each line cut.
- Rich colormap controls with power law scaling, midpoint-aware scaling, and live color range adjustment.
- Built-in menus for rotation, symmetrization, averaging, cropping, coordinate reassignment, Fermi edge correction, and other common operations.
- Tight integration with common analysis helper tools such as {ref}`ktool <guide-ktool>`, {ref}`dtool <guide-dtool>`, and other tools listed in {ref}`interactive-misc-tools`, all accessible from ImageTool’s menus and context menus.
- Seamless integration with {ref}`ImageTool manager <imagetool-manager>` when you need to organize many windows, share workspaces, or synchronize with Jupyter notebooks.

(imagetool-launch)=

## Launching ImageTool

ImageTool expects *image-like* data—usually a {class}`DataArray <xarray.DataArray>` with 2–4 dimensions. ImageTool tries to handle incompatible input dimensions by adding a new dimension for 1D data and squeezing out dimensions of size 1 for 5D+ data. Non-uniform coordinates gain parallel `_idx` indices so you can still slice by position. Supported inputs include numpy arrays, {class}`Dataset <xarray.Dataset>`, or entire {class}`DataTree <xarray.DataTree>` objects; Dataset and DataTree inputs open one ImageTool window per valid variable.

(imagetool-entry-points)=

### Entry points

- Use the {meth}`xarray.DataArray.qshow` accessor on your data:

  ```python
  data.qshow(link=True)
  ```

- Call {func}`erlab.interactive.imagetool.itool` directly:

  ```python
  import erlab.interactive as eri

  eri.itool(data, cmap="cividis")
  ```

  Passing a list or dataset to {func}`itool <erlab.interactive.imagetool.itool>` spawns multiple windows; set `link=True` to synchronize their cursor positions and bins.

- Launch ImageTool from IPython or a notebook using the `%itool` line magic. Load the extension first:

  ```python
  %load_ext erlab.interactive
  ```

  Then run `%itool data`. You can pass additional flags, such as `-m` or `--manager` which sends the window straight to the {ref}`manager <imagetool-manager>`. Type `%itool --help` or `%itool?` to list all supported flags.

- Need to open a file quickly? Use {guilabel}`File → Open…` inside ImageTool. The dialog lists every available loader, including those from data loader plugins, so you can switch between formats without writing code.

(imagetool-ipython)=

### Notebook auto-loading

Add `erlab.interactive` to your [IPython configuration](https://ipython.readthedocs.io/en/stable/config/intro.html) so `%itool` is ready whenever a notebook starts.

:::{tip}

If you use VS Code with the Jupyter extension, add this to your workspace or user `settings.json`:

```json
"jupyter.runStartupCommands": [
    "%load_ext erlab.interactive"
]
```

:::

To integrate ImageTool windows with notebook variables—including bi-directional updates—switch to the {ref}`ImageTool manager <imagetool-manager>` and use the `%watch` magic described in {ref}`working-with-notebooks`.

(imagetool-interface)=

## Interface tour

Every ImageTool window is built from an {class}`ImageSlicerArea <erlab.interactive.imagetool.core.ImageSlicerArea>` plus dockable control panels:

- **Main image and cross-sections** – The central pyqtgraph view renders the current 2D slice. Orthogonal slices and cursor readouts update in real time as you move the cursors.

- **Cursor panel** – Add, remove, and modify cursors here. The coordinates of the active cursor are shown in editable text boxes.

- **Color panel** – Manipulate colormap normalization and appearance.

- **Binning panel** – Set bin widths per dimension and reset them with {material-regular}`settings_backup_restore`. Changes to the bin widths you make while {material-regular}`sync` is toggled are applied to all cursors.

(imagetool-data)=

## Working with dimensions and coordinates

- The order of dimensions can be swapped using the arrow buttons in the cursor panel. The direction of the arrow intuitively indicates which slice will be swapped with the main view.

- Non-uniform coordinates are converted with a `_idx` suffix for plotting. Their true values are displayed in the cursor readouts. ImageTool automatically keeps track of coordinate transformations when you process the data.

- Use {guilabel}`Edit → Coordinate Editor` to manually reassign coordinate values. This is just a GUI for {meth}`xarray.DataArray.assign_coords`, which lets you specify start/end values or per-point overrides.

- Dask-backed arrays are fully supported. The dedicated {guilabel}`Dask` menu exposes actions to compute the array into memory, rechunk automatically, or choose custom chunk shapes within ImageTool.

- Overlay plots of non-dimensional coordinates (e.g., temperature) on the data from {guilabel}`View → Associated Coordinates`.

(imagetool-slicing)=

## Slicing, binning, and linking

- Drag with the left mouse button to pan and with the right mouse button (or the wheel) to zoom. Scroll on individual axes to zoom along a single dimension.

- To change the slicing position, drag on a cursor line to move it, or drag on the plots while holding {kbd}`Ctrl`. You can also use the keyboard shortcuts listed in the {guilabel}`View → Cursor Control` submenu which allow precise nudging of the active cursor with arrow keys.

- When binning is enabled, the data shown are an average over the specified bin widths, which are indicated by shaded regions near the cursor lines.

- When comparing data in several ImageTool windows, you can link them either at creation (`eri.itool([data1, data2], link=True)`) or inside the manager. Linked windows maintain synchronized slicing positions, bin widths, and cursor counts.

(imagetool-cursors)=

## Cursor control and context menus

- Hover over any toolbar icon to see a short description of its function.

- Copy the numeric readouts at any time with {kbd}`Ctrl+Shift+C` (cursor values) or {kbd}`Ctrl+Alt+C` (cursor indices). ImageTool copies native Python literals so you can paste them directly into scripts.

- Multiple cursors can be added to the image using the {material-regular}`add` button in the cursor panel. They can each be dragged independently, and their bin widths can be set separately in the binning panel. To switch the active cursor, simply click on it or select it from dropdown menu in the cursor panel.

- To move all cursors simultaneously, hold {kbd}`Alt` while dragging a cursor line, or use {kbd}`Ctrl+Alt` while dragging on the image.

- Right-click on the image or cut plots to open useful context menus. Common options include copying the slicing code, locking the aspect ratio, exporting the current slice to a matplotlib window or a file, or launching helpers such as {ref}`ktool <guide-ktool>`, {ref}`dtool <guide-dtool>`, {ref}`goldtool <guide-goldtool>`, and {ref}`restool <guide-restool>`.

  :::{hint}
  Holding {kbd}`Alt` while opening the menu switches many actions to cropped mode, which crops the data to what is currently visible in the plot before performing the action. This is useful for conducting analysis on a specific region.
  ::

- Add quick measurement aids by enabling the rotation guideline {guilabel}`View → Rotation Guidelines`. The guideline angles feed directly into the Rotate dialog so you can align your sample quickly.

- The default color cycle of cursors is user configurable. See [](./options.md).

- Colors can be changed individually from {guilabel}`View → Edit Cursor Colors...`, where you can choose from a colormap or edit each cursor's color separately.

- Alternatively, the colors of the cursors can be set to follow a specific coordinate dynamically based on their positions. This can be enabled from {guilabel}`View → Set Cursor Colors by Coordinate...`.

(imagetool-color)=

## Color and normalization

- Toggle {material-regular}`brightness_auto` to lock the color range to the global data min/max and display a colorbar alongside the image. Drag on the colorbar to update limits interactively or right-click to type exact bounds.

- {material-regular}`vertical_align_center` applies gamma scaling relative to the midpoint, which is handy for centered intensity scales such as spin-polarized or dichroic data.

- Use {material-regular}`exposure` to flip between normalization behaviors of {class}`matplotlib.colors.PowerNorm` and {class}`erlab.plotting.colors.InversePowerNorm`.

- By default, only a subset of Matplotlib colormaps is loaded. You can load the whole catalog by right-clicking on the colormap drop-down and selecting {guilabel}`Load All Colormaps`.

(imagetool-editing)=

## Editing and filtering data

Editing dialogs live under the {guilabel}`Edit` and {guilabel}`View` menus. Most transforms are destructive yet provide an {guilabel}`Open in New Window` checkbox so you can keep the original data. When {guilabel}`Copy Code` is available, the generated snippet is placed on your clipboard, ready to paste into a script or notebook for reproducibility.

- **Rotate** – Enter the angle, center, interpolation order, and whether to reshape the image. If a rotation guideline is active the dialog pre-fills the matching angle and pivot.
- **Average Over Dimensions** – Select any set of dimensions to average via {meth}`xarray.DataArray.qsel.average`.
- **Symmetrize** – Mirror a selected dimension about a specified center. Options include additive vs. subtractive symmetry, `valid` vs. `full` overlap, and whether to keep both halves or a single side.
- **Crop** – Enter exact coordinate ranges, or choose {guilabel}`Crop to View` to grab the currently visible extent.
- **Edge Correction** – If your data exposes an `eV` axis, ImageTool can import a previously fitted edge via {func}`xarray_lmfit.load_fit` and shift the spectrum accordingly.
- **Normalize** ({guilabel}`View → Normalize`) – A non-destructive filter that supports area normalization, min-max scaling, and baseline subtraction. You can preview the effect, undo it, or reapply a different normalization later.
- **Coordinate Editor** – Precisely reassign coordinates, including non-uniform axes, via a combination of start/end/delta controls and per-point overrides.

Use {guilabel}`Edit → Undo/Redo` to walk changes back, and {guilabel}`View → Reset` to remove any currently applied filter function. ImageTool also keeps track of additional helper windows opened from the context menus, so everything is closed cleanly when the main window exits.

(imagetool-roi)=

## Regions of Interest (ROIs)

ROIs let you focus on a sub-region of the data. They can be created and manipulated directly on the image plot.

Only polygonal ROIs with arbitrary vertex counts are supported at this time.

### Create and manipulate

- Right-click on an image plot and choose {guilabel}`Add Polygon ROI`. A two-point line appears near the active cursor so you can immediately drag it into place.
- Drag any handle to move a vertex. Click on a segment to insert a new vertex.
- Right-click on a ROI and pick {guilabel}`Edit ROI…` to open a tabular editor.
  - The coordinate table lists every vertex of the ROI.
  - Check the {guilabel}`Closed` option to convert an open polyline into a filled polygon.
- Edits are logged, so you can undo accidental drags with {kbd}`Ctrl+Z`.

### ROI-driven analysis

Two additional context-menu actions appear upon right-clicking on a ROI:

- **Slice Along ROI Path** interpolates the data with {func}`erlab.analysis.interpolate.slice_along_path`.

  Choose a step size and a name for the new path dimension, then decide whether to open the result in a new window or replace the current data.

- **Mask Data with ROI** calls {func}`erlab.analysis.mask.mask_with_polygon` to mask the data with the ROI.

  You can choose whether to invert the mask and whether to trim the resulting data.

Note that both procedures work on the entire data volume, not just the visible slice.

(imagetool-export)=

## Exporting, automation, and preferences

- {guilabel}`File → Save As…` exports the current data to NetCDF or HDF5.

- {guilabel}`File → Move to Manager` hands the window off to the {ref}`ImageTool manager <imagetool-manager>`.

- Keep your configuration consistent across runs via [](./options.md).

(imagetool-shortcuts)=

## Keyboard shortcuts

Most actions advertise their shortcut directly in the menu bar. The table below highlights common gestures. Replace {kbd}`Ctrl` with {kbd}`⌘` and {kbd}`Alt` with {kbd}`⌥` on macOS.

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

Rule of thumb: hold {kbd}`Alt` to apply actions to all cursors. Shortcuts for 'shifting' a cursor involves the {kbd}`Shift` key.
