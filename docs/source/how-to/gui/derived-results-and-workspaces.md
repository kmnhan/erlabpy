# Derived results and workspaces

Use these guides to preserve source data, update derived results, reuse operations, and
manage workspaces.
Read {doc}`workspaces, provenance, and generated code
<../../explanation/workspaces-and-provenance>` before you decide which state to save or
reuse.

(how-to-gui-keep-source-after-transform)=

## Keeping source data during a transformation

Use a managed child result when an ImageTool transformation must preserve the source
data and record how the result was made.

1. Open the source data in ImageTool Manager.
2. Open the required transformation from the ImageTool {guilabel}`Edit` menu.
3. Enter the transformation parameters.
4. Set {guilabel}`Result Placement` to {guilabel}`Open Child Window`.
5. Open the result and inspect it before continuing the analysis.

The Manager places the result below its source and records the operation. Use
{guilabel}`Copy Code` in the dialog or {guilabel}`Copy Full Code` in the Manager to
reproduce it in Python.

Use {guilabel}`Replace Current` only when replacing the current ImageTool is intended.
See {ref}`imagetool-editing` for the available operations and
{ref}`imagetool-manager-result-placement` for all placement states.

(how-to-gui-update-derived-results)=

(imagetool-manager-refresh)=

## Repeating an analysis after an input change

When a source row changes, select its derived row and check the state badge:

- {guilabel}`Stale` means the recorded operation can be repeated with the current
  source.
- {guilabel}`Unavailable` means the current source no longer satisfies the recorded
  selection or operation.
- {guilabel}`Auto` means automatic updates are enabled and the row is current.

Click the badge or the update banner in the tool window. Choose
{guilabel}`Update Now` to repeat the operation once. To change future behavior, enable
or disable automatic updates and choose {guilabel}`Save`. Saving the preference does
not update the current result.

For ftool, goldtool, or restool, enable {guilabel}`Refit after update` only when the same
fit model and settings remain valid for the changed input. Inspect the repeated fit
before using its parameters.

(imagetool-manager-derived-data)=

### Updating a result with several inputs

For a concatenation or console expression made from several ImageTools:

1. Confirm whether the badge is {guilabel}`Changed` or {guilabel}`Missing`.
2. Restore any missing live input or confirm that its recorded source file is available.
3. Choose {guilabel}`Reload Data`.
4. Inspect the recomputed result and its input list.

If a recorded file moved, edit the file-load step and enable
{guilabel}`Also relink selected file loads` to update related inputs together. Only
replay console expressions from workspaces you trust.

Use {ref}`how-to-gui-reuse-manager-operations` when the recorded steps must be applied
to a different open dataset.

(how-to-gui-reuse-manager-operations)=

(imagetool-manager-replay-code)=

## Applying recorded operations to other data

1. Select the derived ImageTool row whose operation sequence must be reused.
2. In the side panel, inspect the recorded inputs and steps.
3. Right-click the step list and copy the required steps.
4. Select the target ImageTool row.
5. Choose {guilabel}`Paste` and inspect the generated result.

Use {guilabel}`Reorder Steps…` before copying when the recorded order must change. A
step can be reused only when the target data contains compatible dimensions,
coordinates, and metadata.

To continue in Python, copy the selected steps or choose {guilabel}`Copy Full Code`.
Inspect file paths, variable names, and operation parameters before running the code.
Code for data loaded from files includes the load operation. Code for watched data uses
the watched variable name.

(how-to-gui-save-manager-workspace)=

(imagetool-manager-workspace)=

## Saving and reopening a workspace

1. In Manager or any managed child window, press {kbd}`Ctrl+S`.
2. Choose a `.itws` path and save the workspace.
3. Confirm the active path with {menuselection}`File --> Workspace Properties`.

The workspace stores managed windows, their state, row notes, derived relationships,
and Figure Composer recipes. Row notes remain workspace metadata and do not modify the
underlying {class}`DataArray <xarray.DataArray>` attributes.

To reopen the session, choose {menuselection}`File --> Open Workspace…`, press
{kbd}`Ctrl+O`, or drag the `.itws` file into Manager. Inspect the restored windows and
derived rows before continuing work.

Use {menuselection}`File --> Add Windows From Workspace…` when selected windows from
another workspace must be added without replacing the current session.

Watched notebook rows reopen disconnected. Recreate their variables and follow
{ref}`imagetool-manager-reconnect-watches` to reconnect them.

(how-to-gui-review-workspace-code)=

## Reviewing stored workspace code

Use this procedure when Manager shows the warning that stored executable content is
paused.

1. Confirm who created the `.itws` file and how you received it.
2. Select {guilabel}`Review and Trust…` in the warning banner. You can also open
   {menuselection}`File --> Workspace Properties` and select
   {guilabel}`Review and Trust…` there.
3. Inspect every item in the review details. The approval applies to the complete
   workspace code listing, not only to the operation that was blocked.
4. If a serialized fitting payload shows only a digest and you cannot verify its
   source, cancel the review.
5. If you accept all listed content, select
   {guilabel}`Trust Workspace and Run Code`.
6. Repeat the blocked action, such as {guilabel}`Reload Data` or a Figure Composer
   render.

If you cancel, data and other non-executable workspace state remain available. Manager
keeps stored code paused.

After a successful save, an unchanged approved workspace normally opens without another
review on the same computer. If its executable content no longer matches the stored
approval, review the workspace again.

See {doc}`workspaces and provenance
<../../explanation/workspaces-and-provenance>` for the content that ERLab pauses. Use
the {ref}`Security settings <options-trusted-workspace-folders>` only for a controlled
analysis folder.

(how-to-gui-reduce-manager-memory)=

## Reducing Manager memory use

1. Save the current Manager session as a workspace.
2. Select the ImageTool rows whose data can be read from the workspace on demand.
3. Choose {guilabel}`Offload to Workspace`.
4. Confirm that the rows show the Dask badge and that their ImageTools still display
   the required slices.

Offloaded data uses less memory but responds more slowly because slices are read from
the workspace file. Keep the workspace file available at its saved path.

To restore selected data to memory, use {menuselection}`Dask --> Load Into Memory` in
ImageTool.
