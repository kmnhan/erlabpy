# Workspaces and provenance

ImageTool Manager keeps data, operation history, and window state. Each item has a
different purpose. A saved workspace can contain them together, but it does not make
them equivalent.

## Data, history, workspaces, and code

| Item | Contents | Main purpose |
| --- | --- | --- |
| `DataArray` | Values, dimensions, coordinates, and metadata | Analysis and display |
| Provenance | Source inputs and operation history for a derived row | Inspection and supported reload |
| Workspace file | Manager rows, data, tools, figures, and window settings | Reopening a Manager session |
| Extension source | Script name, capability ID, source hash, and an optional source copy | Replaying a lab-specific routine or loader |
| Generated code | Python statements for selected operations | Continuing the analysis in a script or notebook |

## Provenance

A derived row appears below its source in the Manager tree. The
{guilabel}`Provenance` tab lists the steps used to create the selected row.

A source change can make a dependent tool or result stale. Manager can repeat a
supported operation when its source remains available and compatible. A missing or
incompatible source prevents that update.

Provenance records how a derived row was made. The values, coordinates, and metadata
remain part of the `DataArray`.

## Workspace limits

A workspace file stores supported Manager data, parent-child rows, tools, figures, and
window settings. Opening the file restores the saved GUI session, subject to the
limits below.

A watched notebook variable remains an external source. Manager can reconnect it by
variable name when the notebook connection is available. Reconnection also depends on
the notebook input data and environment.

A workspace is not a substitute for:

- Original measurement files.
- The software environment.
- Metadata that was not recorded.
- Other variables and Python objects in the notebook kernel.

## Extension scripts

An extension operation records the script name, capability ID, and source hash. The
capability ID identifies the routine or loader. The source hash identifies the exact
script contents that produced the result.

Manager runs an extension operation only when a registered local script is enabled,
approved, and matches the recorded source hash. A source edit therefore requires new
approval. This rule prevents an old provenance record from silently running changed
code.

By default, a workspace embeds the exact source for each extension that its recorded
operations use. The embedded copy is recovery data. Manager does not run it directly.
You must review, save, and register the source before Manager can use it.

Use the {ref}`extension recovery procedure <how-to-gui-recover-extension>` when the
registered script is unavailable. The {ref}`extension reference
<imagetool-manager-extensions>` describes script identity and embedding policies.

## Generated code

Select one or more steps in the {guilabel}`Provenance` tab and choose
{guilabel}`Copy` to copy code for those steps. {guilabel}`Copy Full Code` also includes
supported inputs and preceding steps. The copied code can be reviewed, edited, and run
outside Manager.

Review generated code before you run it. A workspace reopens the GUI session.
Generated code reproduces supported analysis steps in a script or notebook.

Use the {ref}`workspace task guide <how-to-gui-save-manager-workspace>` to save or
restore a session. The {ref}`Manager reference <imagetool-manager-replay-code>`
describes operation history and code generation.
