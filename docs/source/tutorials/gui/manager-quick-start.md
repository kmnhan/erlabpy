# Manager quick start

In this tutorial, you will inspect a simulated ARPES dataset, create a child ImageTool,
inspect its provenance, create a figure, and save the Manager state as a workspace.

You will complete all steps in ImageTool Manager. You do not need a notebook.

## Downloading the tutorial data

[Download the Manager tutorial dataset](../../_static/tutorial-data/manager-quick-start.h5).
Save the file in a location that you can find from a file dialog.

The file contains a float32 DataArray with `alpha`, `beta`, and `eV` dimensions. It is
an HDF5 file written with {meth}`xarray.DataArray.to_netcdf` and
`engine="h5netcdf"`.

## Opening the data

1. Start ImageTool Manager with `itool-manager` or open the standalone application.
2. Choose {menuselection}`File --> Add Data Files…`.
3. Select `manager-quick-start.h5`.
4. If a file-type choice appears, choose {guilabel}`xarray HDF5 Files`.

The Manager adds one top-level ImageTool row named `example_map` and opens its ImageTool
window.

:::{admonition} Author screenshot required
:class: warning

- **Target:** `docs/source/images/tutorials/manager-quick-start-open.png`
- **Capture:** ImageTool Manager with the `example_map` row selected and its ImageTool
  visible beside it.
- **Theme and framing:** Light theme. Show both complete windows. Exclude desktop icons
  and unrelated applications.
- **Caption:** Tutorial data open in ImageTool Manager and ImageTool.
- **Alt text:** ImageTool Manager with one example_map row and its three-dimensional data
  displayed in ImageTool.
:::

## Inspecting the data

Drag the cursor in the main image and cross-sections. The cursor panel shows the
coordinate values. Change a width in the Binning panel and confirm that the shaded
range and displayed average change. Binning changes the displayed slices. It does not
change the underlying DataArray.

## Creating a child ImageTool

1. In ImageTool, choose {menuselection}`Edit --> Select Data…`.
2. Select the `beta` range from `-4` to `4`.
3. Set {guilabel}`Result Placement` to {guilabel}`Open Child Window`.
4. Select {guilabel}`OK`.

The Manager places the new ImageTool below `example_map`. Its provenance records the
source row and the `beta` selection.

Select the child row in the Manager and open the {guilabel}`Provenance` tab. The tab
shows the source and the recorded `sel(...)` step. Use {guilabel}`Copy Full Code` to
inspect the Python code that reproduces the result.

:::{admonition} Author screenshot required
:class: warning

- **Target:** `docs/source/images/tutorials/manager-quick-start-provenance.png`
- **Capture:** Manager tree with the selected child below `example_map` and the
  Provenance tab showing the selection operation.
- **Theme and framing:** Light theme. Capture the Manager window only. Keep the complete
  tree and Provenance tab visible.
- **Caption:** A child ImageTool and its provenance in the Manager.
- **Alt text:** ImageTool Manager showing an example_map parent, a selected child row, and
  the recorded selection step.
:::

## Creating a figure

1. Keep the child ImageTool selected in the Manager.
2. Open its context menu and choose {guilabel}`Add to Figure…`.
3. If another Figure Composer is already open, set {guilabel}`Action` to
   {guilabel}`New Figure`.

Figure Composer adds the selected data as a source and creates an initial plotting
step. The Manager also adds a {guilabel}`Figures` tab.

:::{admonition} Author screenshot required
:class: warning

- **Target:** `docs/source/images/tutorials/manager-quick-start-figure.png`
- **Capture:** Figure Composer with the tutorial source, recipe, and rendered figure
  visible.
- **Theme and framing:** Light theme. Capture the complete Figure Composer controls and
  figure window.
- **Caption:** A Figure Composer recipe created from the tutorial data.
- **Alt text:** Figure Composer showing the tutorial data source, plotting recipe, and
  rendered image.
:::

## Saving the workspace

Choose {menuselection}`File --> Save Workspace As…` in the Manager and save an `.itws`
file. The workspace stores the source and child ImageTools, their provenance, and the
Figure Composer source and recipe.

You have opened data, created a child ImageTool, inspected its provenance, created a
figure, and saved a workspace. Read {doc}`Python and GUI workflows
<../../explanation/python-and-gui-workflows>` for the role of each application. Read
{doc}`workspaces, provenance, and generated code
<../../explanation/workspaces-and-provenance>` before you reuse or share this work.

Use the {doc}`GUI how-to guides <../../how-to/gui/index>` for specific tasks. Use the
{doc}`GUI reference <../../reference/gui/index>` to look up a control.
