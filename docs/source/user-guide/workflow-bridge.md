(workflow-bridge)=

# GUI and Python Side by Side

ERLabPy has two main types of workflows:

- Python code-based workflows in scripts and notebooks.
- GUI workflows in {ref}`ImageTool <imagetool-interface>`, the {ref}`ImageTool manager <imagetool-manager>`, and other {ref}`interactive tools <interactive-misc-tools>`.

The bridge between the two is built into the package. GUI actions either map directly to
public Python APIs or can even generate reproducible code that is ready to paste back
into a notebook. This design allows you to fluidly move between the GUI and code, using
each where it is most effective.

Use the GUI to discover selections and parameters, then move the final steps back into
code for reproducibility and batch processing.

(workflow-bridge-operations)=

## Operation map

:::{list-table}
:header-rows: 1

- - Analysis task
  - GUI entry point
  - Python/API equivalent
- - Open a DataArray in ImageTool
  - {ref}`ImageTool entry points <imagetool-entry-points>`
  - {meth}`xarray.DataArray.qshow`, {func}`erlab.interactive.imagetool.itool`,
    `%itool data`
- - Browse a data directory
  - {ref}`Data Explorer <guide-data-explorer>`
  - {func}`erlab.io.load`, {func}`erlab.io.set_loader`, {func}`erlab.io.set_data_dir`
- - Open or replace data in the manager
  - {ref}`Manager opening and replacement paths <imagetool-manager-open>`
  - `data.qshow(manager=True)` and `eri.itool(data, manager=True)`, see
    {ref}`working-with-notebooks` for notebook-manager synchronization options.
- - Select a point or range from multidimensional data
  - {guilabel}`Edit → Select Data…` or the right-click context menu of each plot
  - {meth}`xarray.DataArray.qsel`, {meth}`xarray.DataArray.sel`, and
    {meth}`xarray.DataArray.isel`
- - Aggregate over dimensions
  - {ref}`Aggregation dialog <imagetool-editing>`
  - {meth}`xarray.DataArray.qsel.mean`, {meth}`~xarray.DataArray.qsel.min`,
    {meth}`~xarray.DataArray.qsel.max`, and {meth}`~xarray.DataArray.qsel.sum`
- - Interpolate along a dimension
  - {ref}`Interpolation dialog <imagetool-editing>`
  - {meth}`xarray.DataArray.interp`
- - Coarsen data
  - {ref}`Coarsen dialog <imagetool-editing>`
  - {meth}`xarray.DataArray.coarsen` followed by a reducer such as `.mean()` or
    `.sum()`
- - Thin data
  - {ref}`Thin dialog <imagetool-editing>`
  - {meth}`xarray.DataArray.thin`
- - Crop data
  - {ref}`Crop dialogs <imagetool-editing>`
  - {meth}`xarray.DataArray.sel`, {meth}`xarray.DataArray.isel`
- - Rotate or symmetrize
  - {ref}`Transform dialogs <imagetool-editing>` including {guilabel}`Edit → Rotate` and
    {guilabel}`Edit → Symmetrize`
  - {func}`erlab.analysis.transform.rotate`,
    {func}`erlab.analysis.transform.symmetrize`,
    {func}`erlab.analysis.transform.symmetrize_nfold`
- - Edit or add coordinates
  - {guilabel}`Edit → Edit Coordinates` in {ref}`ImageTool data controls <imagetool-data>`
  - {meth}`xarray.DataArray.assign_coords`, such as
    `data.assign_coords(y=scale * data.y + offset)`,
    `data.assign_coords(temperature=20.0)`, or
    `data.assign_coords(label=("x", labels))`
- - Edit or add attributes
  - {guilabel}`Edit → Edit Attributes` in {ref}`ImageTool data controls <imagetool-data>`
  - {meth}`xarray.DataArray.assign_attrs`, such as
    `data.assign_attrs(sample_temp=20.0, note="checked")`
- - Swap dimensions
  - {ref}`Coordinate editing dialog <imagetool-data>`
  - {meth}`xarray.DataArray.swap_dims`
- - Normalize interactively
  - {ref}`ImageTool normalization dialog <imagetool-editing>`
  - Expressions like `data / data.mean(...)`
- - Combine manager ImageTools
  - {guilabel}`Concatenate` in the {ref}`ImageTool manager <imagetool-manager-derived-data>`
  - {func}`xarray.concat`
- - Slice along an ROI path
  - {ref}`ROI context menu <imagetool-roi>`
  - {func}`erlab.analysis.interpolate.slice_along_path`
- - Mask with an ROI
  - {ref}`ROI context menu <imagetool-roi>`
  - {func}`erlab.analysis.mask.mask_with_polygon`
- - Momentum conversion
  - {ref}`ktool <guide-ktool>` (Can be launched from the View menu)
  - {meth}`xarray.DataArray.kspace.convert`
- - Interactive fitting
  - {ref}`ftool <guide-ftool>`, {ref}`goldtool <guide-goldtool>`, and {ref}`restool <guide-restool>`
    (Can be launched from the {ref}`Plot context menu <imagetool-cursors>`)
  - {meth}`xarray.DataArray.xlm.modelfit` and scripted fitting workflows in
    {mod}`erlab.analysis.gold`
- - Pull data back out of the manager
  - {ref}`Manager notebook bridge <working-with-notebooks>`
  - {func}`erlab.interactive.imagetool.manager.fetch`, `%watch`, `%store`
:::
