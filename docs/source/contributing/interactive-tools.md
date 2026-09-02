(interactive-tool-authoring)=

# Authoring interactive tools

Use this guide to add an interactive analysis tool to {mod}`erlab.interactive`. It
assumes that you can write Qt widgets and work with {class}`xarray.DataArray` objects.
See {doc}`development` for repository setup and contribution checks.

Use {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` when a window analyzes a
{class}`~xarray.DataArray` and must support save and restore, undo and redo, or
ImageTool Manager integration. A plain Qt window is sufficient for an internal utility
that does not need these features.

See the {ref}`Interactive-tool authoring API <interactive-tool-authoring-api>` for the
protected extension-point signatures.

## Required tool interface

A {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` subclass must provide these
members:

| Member | Requirement |
| --- | --- |
| Constructor | Accept the primary `data` array as the first and only required positional argument. |
| {attr}`tool_name <erlab.interactive.utils.ToolWindow.tool_name>` | Use a stable, short name. |
| {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>` | Store the lightweight widget state required to restore the window. |
| {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>` | Return the primary input array. |
| {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` | Return and apply a complete {attr}`~erlab.interactive.utils.ToolWindow.StateModel`. |

Define the class at module scope. Saved tools store its module and qualified name.

## Minimal implementation

The following tool displays a scaled two-dimensional array. It has persistent state and
undo history, but no optional Manager features.

```python
from __future__ import annotations

import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtWidgets, QtCore

import erlab


class ScaleTool(erlab.interactive.utils.ToolWindow):
    tool_name = "scaletool"

    class StateModel(pydantic.BaseModel):
        scale: float = 1.0

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = self._validate_data(data)

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        self.setCentralWidget(root)

        self.plot = pg.PlotWidget()
        self.image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 100.0)
        self.scale_spin.setValue(1.0)

        self.plot.addItem(self.image)
        layout.addWidget(self.plot)
        layout.addWidget(self.scale_spin)

        self.scale_spin.valueChanged.connect(self._scale_changed)
        self._refresh()
        self._reset_history_stack()

    @staticmethod
    def _validate_data(data: xr.DataArray) -> xr.DataArray:
        parsed = erlab.interactive.utils.parse_data(data)
        if parsed.ndim != 2:
            raise ValueError("`data` must be two-dimensional")
        return parsed

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(scale=float(self.scale_spin.value()))

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        with QtCore.QSignalBlocker(self.scale_spin):
            self.scale_spin.setValue(status.scale)
        self._refresh()

    def _scale_changed(self, _value: float) -> None:
        self._refresh()
        self._write_state()

    def _refresh(self) -> None:
        self.image.setDataArray(self.tool_data * self.scale_spin.value())
```

Use {meth}`ToolWindow.setCentralWidget()
<erlab.interactive.utils.ToolWindow.setCentralWidget>`, not the base
`QMainWindow` implementation. The override retains the status area used for source
updates.

Call {meth}`_write_state() <erlab.interactive.utils.ToolWindow._write_state>` after an
undoable user action. Do not call it from the
{attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` setter. The setter
can update dependent displays, but it must not record a user action.

Reset history after the initial widget state is complete. The saved state model must
contain enough information for its setter to reproduce the visible tool state.

Add a public launcher beside the class:

```python
def scaletool(
    data: xr.DataArray, *, execute: bool | None = None
) -> ScaleTool:
    """Open the scale tool."""
    with erlab.interactive.utils.setup_qapp(execute):
        window = ScaleTool(data)
        window.show()
        window.raise_()
        window.activateWindow()
    return window
```

For example:

```python
import erlab
from erlab.io.exampledata import generate_data_angles

data = generate_data_angles().qsel(beta=0.0)
window = scaletool(data)
```

## State notifications

Use the narrowest notification that describes the change.

| Change | Action |
| --- | --- |
| An undoable control changes | Call {meth}`_write_state() <erlab.interactive.utils.ToolWindow._write_state>`. |
| A saved setting changes without an undo step | Emit {attr}`sigStateChanged <erlab.interactive.utils.ToolWindow.sigStateChanged>`. |
| Only {attr}`preview_imageitem <erlab.interactive.utils.ToolWindow.preview_imageitem>` or {attr}`info_text <erlab.interactive.utils.ToolWindow.info_text>` changes | Emit {attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>`. |
| Displayed data or a Manager-visible output changes | Call {meth}`_notify_data_changed() <erlab.interactive.utils.ToolWindow._notify_data_changed>`. |

{meth}`_notify_data_changed() <erlab.interactive.utils.ToolWindow._notify_data_changed>`
updates Manager information and output data. During a source refresh, the base class
delays the data-change signal until the complete input update succeeds.

## Input updates

Implement input updates only when an existing window must follow data from ImageTool.
The framework passes one complete mapping of declared input names.

```python
from collections.abc import Mapping


def validate_update_inputs(
    self, inputs: Mapping[str, xr.DataArray]
) -> Mapping[str, xr.DataArray]:
    validated = dict(super().validate_update_inputs(inputs))
    validated["data"] = self._validate_data(validated["data"])
    return validated


def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
    status = self.tool_status
    self._data = inputs["data"]
    with self._history_suppressed():
        self.tool_status = status
    self._reset_history_stack()
    return True
```

The update must modify the existing window. Do not replace it with a new window.
Return the same input names from `validate_update_inputs()`. The framework rejects
missing or additional names. Return `False` from `update_inputs()` if the input was not
applied.

For a tool with background work, override
{meth}`_cancel_background_work()
<erlab.interactive.utils.ToolWindow._cancel_background_work>` so an update cannot
replace data while an old worker still uses it.

If an update finishes asynchronously, use this sequence:

1. Call {meth}`_defer_source_refresh()
   <erlab.interactive.utils.ToolWindow._defer_source_refresh>` before
   `update_inputs()` returns `False`.
2. Apply the result only if it still belongs to the current request.
3. Call {meth}`finalize_source_refresh()
   <erlab.interactive.utils.ToolWindow.finalize_source_refresh>` after publication.
4. Call {meth}`abort_source_refresh()
   <erlab.interactive.utils.ToolWindow.abort_source_refresh>` if the work stops before
   publication.

See {class}`GoldTool <erlab.interactive.fermiedge.GoldTool>` for an asynchronous
implementation.

## Persistence

{meth}`to_dataset() <erlab.interactive.utils.ToolWindow.to_dataset>` stores
{attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>` separately from
{attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`. Keep arrays,
fit results, and other large values out of the state model so history snapshots stay
small.

Use extra persistence hooks only when the default data and state are insufficient.

| Requirement | Hooks |
| --- | --- |
| Additional {class}`~xarray.DataArray` inputs or results | {meth}`_persistence_data_items() <erlab.interactive.utils.ToolWindow._persistence_data_items>` and {meth}`_restore_persistence_data_items() <erlab.interactive.utils.ToolWindow._restore_persistence_data_items>` |
| Other save-only values | {meth}`_append_persistence_payload() <erlab.interactive.utils.ToolWindow._append_persistence_payload>` and {meth}`_restore_persistence_payload() <erlab.interactive.utils.ToolWindow._restore_persistence_payload>` |
| Optional expensive restore work | {meth}`_run_or_defer_restore_work() <erlab.interactive.utils.ToolWindow._run_or_defer_restore_work>` |

When you override `_persistence_data_items()`, retain the items returned by `super()`.
Add each additional input under its declared input name. A multi-input tool must also
be constructible from its primary `data` argument alone so saved state can restore the
remaining inputs.

Direct {meth}`from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>` calls
restore optional work immediately. Manager workspace loading can defer work for hidden
tools. Put validation and state required for a usable object outside the deferred
callback.

Do not start optional restore calculations unconditionally in the constructor. A
queued callback can be discarded if the user closes the hidden tool before it is
needed.

## Manager presentation

These properties are optional:

- {attr}`preview_imageitem <erlab.interactive.utils.ToolWindow.preview_imageitem>`
  supplies the Manager thumbnail.
- {attr}`info_text <erlab.interactive.utils.ToolWindow.info_text>` supplies a short HTML
  summary for the Details view.

Emit {attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>` when
either value changes. Use
{meth}`_notify_data_changed() <erlab.interactive.utils.ToolWindow._notify_data_changed>`
instead when output data also changes.

## Generated code

Set {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>` when
the tool has a user-facing **Copy Code** action. Prefer one expression and assignment
for one analysis operation. Add these members to the tool class:

```python
import typing


class ScaleTool(erlab.interactive.utils.ToolWindow):
    # Keep the required members from the minimal implementation.
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from the current scaletool input",
            label="Scale the data",
            expression_method="_scale_expression",
            assign="scaled",
        )
    )

    def _scale_expression(
        self,
        *,
        primary_input: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        del data
        source = primary_input or "data"
        return f"{source} * {self.scale_spin.value()!r}"
```

Connect the visible button to
{meth}`copy_code() <erlab.interactive.utils.ToolWindow.copy_code>`. The declaration
defines the code. It does not create a button.

Generated code is user-facing Python. Use public APIs and meaningful variable names.
Test the code by executing it and comparing the result. Do not test only the string
format.

Use `operations_method` only when one expression cannot represent the action. See
{class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` for this case.

## ImageTool outputs

Declare an output when an ImageTool opened by the tool must remain attached to it in
the Manager and refresh with it. Add these members to the tool class:

```python
import enum
import typing


class ScaleTool(erlab.interactive.utils.ToolWindow):
    # Keep the required members and _scale_expression() from the earlier examples.
    class Output(enum.StrEnum):
        SCALED = "scaletool.scaled"

    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        Output.SCALED: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_scaled_data",
            provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from the current scaletool input",
                label="Scale the data",
                expression_method="_scale_expression",
                assign="scaled",
            ),
        )
    }

    def _scaled_data(self) -> xr.DataArray:
        return self.tool_data * self.scale_spin.value()

    def open_scaled_data(self) -> None:
        self._launch_output_imagetool(
            self._scaled_data(),
            output_id=self.Output.SCALED,
        )
```

Use a stable, tool-qualified output ID. Workspace files store this value. The output
data and its provenance must describe the same calculation.

Declaring an output does not create an action. Connect a button or menu action to the
method that calls
{meth}`_launch_output_imagetool()
<erlab.interactive.utils.ToolWindow._launch_output_imagetool>`.

For an independent ImageTool window, use
{meth}`_launch_detached_output_imagetool()
<erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool>`. If that window
needs generated code, pass the result of
{meth}`detached_output_imagetool_provenance()
<erlab.interactive.utils.ToolWindow.detached_output_imagetool_provenance>` explicitly.
The launch helper does not call the provenance hook.

See {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>` for several declared
outputs.

## Public integration

For a built-in ERLabPy tool:

1. Add the runtime module under `src/erlab/interactive/`.
2. Export the public launcher from `src/erlab/interactive/__init__.pyi`.
3. Add an IPython line magic in `src/erlab/interactive/_magic.py` when notebook users
   need one.
4. Add ImageTool or Manager actions only where the tool applies to the visible data.
5. Add the public launcher to the API reference. Add user documentation for the tasks
   that the tool supports.

An external package must register each restorable tool class:

```toml
[project.entry-points."erlab.interactive.tool_windows"]
scale-tool = "my_package.tools:ScaleTool"
```

The entry-point value must match the class module and qualified name. A class defined
only in a notebook or script cannot be discovered by a fresh Manager process.

## Tests

The core test must cover state and data restoration:

```python
def test_scale_tool_roundtrip(qtbot):
    data = xr.DataArray(
        [[0.0, 1.0], [2.0, 3.0]],
        dims=("alpha", "eV"),
        name="example",
    )
    tool = ScaleTool(data)
    qtbot.addWidget(tool)
    tool.scale_spin.setValue(2.5)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)

    assert isinstance(restored, ScaleTool)
    assert restored.tool_status == tool.tool_status
    xr.testing.assert_identical(restored.tool_data, data)
```

Add only the tests that match the features that the tool implements:

| Feature | Required coverage |
| --- | --- |
| Source updates | Validate and apply the complete input mapping. Confirm that the existing window and its settings remain valid. |
| Background work | Cover cancellation, stale results, failure, and close behavior. Test both Qt bindings for lifetime-sensitive code. |
| Extra persisted values | Round-trip each persistence hook. |
| Generated code | Execute the code in an explicit namespace and compare the result with the tool output. |
| Declared ImageTool outputs | Cover opening, refreshing, persistence, and matching generated code. |
| Dialogs | Cover accepted and cancelled paths. |

If you add a top-level test module, update `scripts/_ci_test_groups.py`. See
{doc}`development` for the full test and pull-request workflow.

## Production examples

Use current tools as implementation references:

| Requirement | Example |
| --- | --- |
| Synchronous calculations and source updates | {class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` |
| Coordinate conversion and a declared ImageTool output | {class}`KspaceTool <erlab.interactive.kspace.KspaceTool>` |
| Asynchronous work and source refresh | {class}`GoldTool <erlab.interactive.fermiedge.GoldTool>` |
| Multiple inputs and several declared outputs | {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>` |

The launcher functions are the public user interfaces. Treat the classes above as
implementation examples.
