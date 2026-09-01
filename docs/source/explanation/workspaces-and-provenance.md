# Workspaces and provenance

ImageTool Manager records GUI operations and can save the GUI session as a workspace.

## Provenance

When a supported GUI operation creates new data, Manager groups the result below its
source and records the operation. The {guilabel}`Provenance` tab shows the recorded
inputs and steps.

The record supports updates and reuse only while the required inputs and operation
remain available and compatible. Provenance describes how the result was created. The
values, coordinates, and metadata remain part of the `DataArray`.

Use {ref}`how-to-gui-update-derived-results` to repeat an operation after an input
changes. Use {ref}`how-to-gui-reuse-manager-operations` to apply recorded operations to
other data.

## Workspace limitations

A workspace restores supported Manager session state. It does not restore the complete
analysis environment.

A watched notebook variable remains an external source and reopens disconnected.

A workspace is not a substitute for:

- Original measurement files.
- The software environment.
- Metadata that was not recorded.
- Other variables and Python objects in the notebook kernel.

Use {ref}`how-to-gui-save-manager-workspace` to save or reopen a Manager session.

## Stored executable content

Some workspaces contain Python used by recorded operations, Figure Composer, or fitting
tools. Manager keeps unverified code paused while it restores the data and other
non-executable session state. Review stored code only when you trust its source.

Use {ref}`how-to-gui-review-workspace-code` when code is paused.
See {ref}`imagetool-manager-code-trust` for the exact paused-code behavior and status
values.

## Workspaces and Python code

A workspace reopens the GUI session. Generated Python code carries supported analysis
operations into a script or notebook. You can review and edit that code independently
of the workspace.

Use {ref}`how-to-gui-reuse-manager-operations` to copy recorded steps or full analysis
code. {ref}`imagetool-manager-replay-code` describes the available code-generation
controls.
