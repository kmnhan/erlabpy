(workflow-bridge)=

# GUI and Python Side by Side

ERLabPy exposes the same analysis pipeline through two entry points:

- Python code-based workflows in scripts and notebooks.
- GUI workflows in ImageTool, the ImageTool manager, and the specialized interactive
  tools, where you discover parameters interactively.

The bridge between them is built into the package. GUI actions either map directly to
public Python APIs, generate reproducible code, or synchronize with notebook variables
through the manager. Use the GUI to discover selections and parameters, then move the
final steps back into code for reproducibility and batch processing.

(workflow-bridge-operations)=

## Operation map

:::{list-table}
:header-rows: 1

- - Analysis task
  - GUI entry point
  - Python/API equivalent
  - Reproducibility note
- - Open a DataArray in ImageTool
  - {ref}`ImageTool entry points <imagetool-entry-points>`
  - {meth}`xarray.DataArray.qshow`, {func}`erlab.interactive.imagetool.itool`,
    `%itool data`
  - Use this as the simplest notebook-to-GUI hand-off for one-off inspection.
- - Browse a data directory
  - {ref}`Data Explorer <guide-data-explorer>`
  - {func}`erlab.io.load`, {func}`erlab.io.set_loader`, {func}`erlab.io.set_data_dir`
  - The Data Explorer is a visual alternative to scripted loading. Use it to quickly
    browse and open files, then write explicit load calls for reproducibility.
- - Open or replace data in the manager
  - {ref}`Manager opening and replacement paths <imagetool-manager-open>`
  - `data.qshow(manager=True)`, `eri.itool(data, manager=True)`
  - Use `%watch` or {func}`erlab.interactive.imagetool.manager.watch` when variable
    updates in the notebook should automatically sync to the manager or vice versa. See
    {ref}`working-with-notebooks` for details.
- - Average over dimensions
  - {ref}`Averaging dialog <imagetool-editing>`
  - {meth}`xarray.DataArray.qsel.average`
  - The dialog maps directly to the accessor call.
- - Crop data
  - {ref}`Crop dialogs <imagetool-editing>`
  - {meth}`xarray.DataArray.sel`, {meth}`xarray.DataArray.isel`
  - Use {guilabel}`Copy selection code` or dialog {guilabel}`Copy Code` to freeze the
    exact bounds chosen in the GUI.
- - Rotate or symmetrize
  - {ref}`Transform dialogs <imagetool-editing>`
  - {func}`erlab.analysis.transform.rotate`,
    {func}`erlab.analysis.transform.symmetrize`
  - Both dialogs generate public API calls with the current parameters.
- - Reassign coordinates
  - {ref}`Coordinate editing dialog <imagetool-data>`
  - {meth}`xarray.DataArray.assign_coords`
  - Use the GUI to refine coordinate values, then keep the final assignment in code.
- - Normalize interactively
  - {ref}`ImageTool normalization dialog <imagetool-editing>`
  - Expressions like `data / data.mean(...)`
  - Treat the GUI as a preview surface, then write the chosen normalization explicitly in
    the notebook.
- - Slice along an ROI path
  - {ref}`ROI context menu <imagetool-roi>`
  - {func}`erlab.analysis.interpolate.slice_along_path`
  - The copied code captures both the vertices and the chosen path step size.
- - Mask with an ROI
  - {ref}`ROI context menu <imagetool-roi>`
  - {func}`erlab.analysis.mask.mask_with_polygon`
  - The copied code captures vertices, dimensions, and mask options.
- - Momentum conversion
  - {ref}`ktool <guide-ktool>` (Can be launced from the View menu)
  - {meth}`xarray.DataArray.kspace.convert`
  - {ref}`ktool <guide-ktool>` is the parameter-discovery layer; {guilabel}`Copy code`
    turns the final settings into code.
- - Interactive fitting
  - {ref}`ftool <guide-ftool>`, {ref}`goldtool <guide-goldtool>`, and {ref}`restool <guide-restool>`
    (Can be launched from the {ref}`Plot context menu <imagetool-cursors>`)
  - {meth}`xarray.DataArray.xlm.modelfit` and scripted fitting workflows in
    {mod}`erlab.analysis.gold`
  - Use the interactive tools to tune parameters, then copy the fit code or save the fit
    result for batch reuse.
- - Pull data back out of the manager
  - {ref}`Manager notebook bridge <working-with-notebooks>`
  - {func}`erlab.interactive.imagetool.manager.fetch`, `%watch`, `%store`
  - The manager is a bridge, not a dead end: fetch copies for scripts, watch for live
    notebook synchronization.
:::
