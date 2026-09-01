(interactive-tool-authoring)=

# Authoring interactive tools

This section is aimed at experienced users who are already comfortable with Qt,
{mod}`xarray`, and ERLabPy's analysis model, and want to contribute a new interactive
tool to {mod}`erlab.interactive`.

It focuses on ERLab-specific integration points: the
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` class, manager support, updates
when ImageTool data changes, public launch paths, and the test/docs work expected in a
contribution. For general repository conventions, see the
{doc}`../contributing`.

For the user-facing workflow, start with {ref}`imagetool-manager-nested-results`,
{ref}`imagetool-manager-result-placement`, {ref}`imagetool-manager-refresh`, and
{ref}`imagetool-manager-replay-code`. This page explains how tool authors make those
features work.

## Start with the right shape

Most user-facing ERLabPy GUIs should inherit from
{class}`erlab.interactive.utils.ToolWindow` for it to correctly integrate with the
{ref}`ImageTool manager <imagetool-manager>`.

In practice, {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` enables several
things:

- save/restore support through
  {meth}`to_dataset() <erlab.interactive.utils.ToolWindow.to_dataset>`,
  {meth}`from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>`,
  {meth}`to_file() <erlab.interactive.utils.ToolWindow.to_file>`, and
  {meth}`from_file() <erlab.interactive.utils.ToolWindow.from_file>`, using
  {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>`,
  {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>`,
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`, and optional
  save-only payload hooks for persisted data that should stay out of undo/redo history;
- standard undo/redo actions for the lightweight UI state stored in
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`;
- integration with the ImageTool manager, including tool naming, preview images, rich
  info text, and manager refresh notifications through
  {attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>`;
- remembering which ImageTool inputs opened the tool, including saved
  metadata, stale or unavailable status tracking, and the built-in update dialog;
- ImageTool windows declared in
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` that
  appear as child rows of the tool in the ImageTool manager and can be reopened,
  refreshed, and persisted; and
- the update hooks used by tools that can react when ImageTool inputs change:
  {meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`,
  {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>`, and
  {meth}`_cancel_background_work() <erlab.interactive.utils.ToolWindow._cancel_background_work>`.

Use {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` when your tool should do
any of the following:

- accept an {class}`xarray.DataArray` as its main input;
- serialize and restore itself through
  {meth}`to_dataset() <erlab.interactive.utils.ToolWindow.to_dataset>` and
  {meth}`from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>`;
- appear as a child row of an ImageTool in the {ref}`ImageTool manager <imagetool-manager>`; or
- refresh itself when that ImageTool's data changes.

{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` assumes a few things about
your implementation:

- The constructor accepts `data` as its primary positional input. Additional initial
  inputs and options can be keyword arguments. The Manager records durable input
  bindings separately. A saveable multi-input tool must let restore omit non-primary
  constructor inputs; restore supplies them through one complete `update_inputs()` call.
- The nested {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>` contains
  the lightweight UI state that participates in undo/redo history.
- {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>` returns the main
  {class}`xarray.DataArray`.
- {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` serializes and
  reapplies the live widget state.
- If the tool needs extra persisted state that should not participate in history,
  override
  {meth}`_append_persistence_payload() <erlab.interactive.utils.ToolWindow._append_persistence_payload>`
  and
  {meth}`_restore_persistence_payload() <erlab.interactive.utils.ToolWindow._restore_persistence_payload>`.

As a practical authoring checklist:

- Required for the core {class}`ToolWindow <erlab.interactive.utils.ToolWindow>`
  interface: constructor with `data`,
  {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>`,
  {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>`, and
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`.
- Required if the tool can be refreshed from ImageTool data:
  {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>`. It
  receives the complete named input mapping, including for a one-input tool. Return
  `False` when the tool does not apply the input. Call `_defer_source_refresh()` first
  if accepted asynchronous work will apply it later.
- Optional, but strongly recommended for user-facing tools:
  {attr}`tool_name <erlab.interactive.utils.ToolWindow.tool_name>` (the base class
  default is just `"tool"`).
- Optional for tools with expensive or bulky save-only state:
  {meth}`_append_persistence_payload() <erlab.interactive.utils.ToolWindow._append_persistence_payload>`
  and
  {meth}`_restore_persistence_payload() <erlab.interactive.utils.ToolWindow._restore_persistence_payload>`.
- Optional manager / provenance integration:
  {meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`,
  {meth}`_cancel_background_work() <erlab.interactive.utils.ToolWindow._cancel_background_work>`,
  {attr}`preview_imageitem <erlab.interactive.utils.ToolWindow.preview_imageitem>`,
  {attr}`info_text <erlab.interactive.utils.ToolWindow.info_text>`,
  {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>`,
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>`, and
  {meth}`detached_output_imagetool_provenance() <erlab.interactive.utils.ToolWindow.detached_output_imagetool_provenance>`.

If your tool is a quick internal prototype or does not need save/restore support, a
plain Qt widget may be enough. For anything that should behave like
{func}`dtool <erlab.interactive.dtool>`, {func}`ftool <erlab.interactive.ftool>`,
{func}`goldtool <erlab.interactive.goldtool>`, or
{func}`ktool <erlab.interactive.ktool>`, start from
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>`.

## Map capabilities to hooks

When you add a new tool, think in terms of user-visible capabilities first and then wire
up the corresponding {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` surface:

- Save and restore the tool window: required; implement
  {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>`,
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`, and
  {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>`. If save/load also
  needs large arrays calculated by the tool, keep them out of
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` and use
  {meth}`_append_persistence_payload() <erlab.interactive.utils.ToolWindow._append_persistence_payload>`
  and
  {meth}`_restore_persistence_payload() <erlab.interactive.utils.ToolWindow._restore_persistence_payload>`
  instead.
- Show rich metadata in the ImageTool manager:
  optional; implement {attr}`info_text <erlab.interactive.utils.ToolWindow.info_text>`
  and
  {attr}`preview_imageitem <erlab.interactive.utils.ToolWindow.preview_imageitem>`, and
  emit {attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>` when
  either changes.
- Refresh the tool when the ImageTool that opened it changes:
  {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>` is part of the
  minimal tool surface;
  {meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`
  and
  {meth}`_cancel_background_work() <erlab.interactive.utils.ToolWindow._cancel_background_work>`
  are optional additions when normalization or worker shutdown matter.
- Generate code that repeats the tool's main action:
  optional; usually set
  {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>` to a
  {class}`ToolScriptProvenanceDefinition <erlab.interactive.utils.ToolScriptProvenanceDefinition>`.
  Prefer {attr}`label <erlab.interactive.utils.ToolScriptProvenanceDefinition.label>`,
  {attr}`expression_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.expression_method>`,
  and {attr}`assign <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign>` for
  the common single-step case. Add
  {attr}`prelude_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.prelude_method>`
  when the generated code needs setup statements before the final expression. Use
  {attr}`operations_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.operations_method>`
  only when the copied script truly needs multiple labeled operations.
- Expose ImageTool windows as child rows of the tool and let the manager update them later:
  optional; declare a stable output id, preferably with {class}`enum.StrEnum`, and add
  it to
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` with
  a
  {class}`ToolImageOutputDefinition <erlab.interactive.utils.ToolImageOutputDefinition>`.
  Its {attr}`data_method <erlab.interactive.utils.ToolImageOutputDefinition.data_method>`
  field should name a zero-argument instance method that returns the current output
  {class}`xarray.DataArray`. Its
  {attr}`provenance <erlab.interactive.utils.ToolImageOutputDefinition.provenance>`
  field defines the code that recreates the output. The string output id is what the
  manager persists in saved workspaces, so use a tool-qualified name such as
  `"mytool.filtered"`.
- Open an ImageTool that is not one of the declared outputs:
  optional; call
  {meth}`_launch_detached_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool>`.
  In the manager, that opens a fresh independent top-level ImageTool window with no
  saved parent row or output id. Outside the manager, each call opens a new standalone
  ImageTool window. When the new window needs replay code, call
  {meth}`detached_output_imagetool_provenance() <erlab.interactive.utils.ToolWindow.detached_output_imagetool_provenance>`
  and pass its result explicitly as `provenance_spec`.
  {meth}`_launch_detached_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool>`
  does not invoke that hook implicitly. Override the hook only when detached launches
  should show different generated code from
  {meth}`current_provenance_spec() <erlab.interactive.utils.ToolWindow.current_provenance_spec>`.
  Keep the hook free of blocking side effects such as modal warnings.

The important distinction is that outputs declared in
{attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` become
ImageTool windows as child rows of the tool in the manager, keyed by a serialized
`output_id`. Any ImageTool opened without an `output_id` is not reproducible as one of
those windows. Do not use
{meth}`_launch_detached_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool>`
as a substitute for a real declared ImageTool output.

In user-facing terms, this is the difference between an ImageTool window kept with the
tool that made it and a detached top-level window; see
{ref}`imagetool-manager-result-placement`.

A real example is {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>`:

- {attr}`Fit2DTool.Output.PARAMETER_VALUES <erlab.interactive._fit2d.Fit2DTool.Output.PARAMETER_VALUES>`
  and
  {attr}`Fit2DTool.Output.PARAMETER_STDERR <erlab.interactive._fit2d.Fit2DTool.Output.PARAMETER_STDERR>`
  are declared in
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>`, so
  those parameter plots become ImageTool
  windows as child rows of the fit tool. Each persisted `output_id` records both
  the output kind and the selected parameter, so refreshing the child window keeps it
  tied to that parameter even if the fit tool's parameter plot later shows another
  parameter.
- {meth}`Fit2DTool._show_dataarray_in_itool() <erlab.interactive._fit2d.Fit2DTool._show_dataarray_in_itool>`
  also has a generic path for arbitrary {class}`xarray.DataArray` objects that are not
  declared outputs. In the manager, that path opens a fresh
  independent top-level ImageTool window each time. Outside the manager, it opens a
  fresh standalone ImageTool window each time.
- That generic path is intentionally not a declared ImageTool output because the manager
  cannot recreate it from either
  {attr}`source_spec <erlab.interactive.utils.ToolWindow.source_spec>` or `output_id`.

## Build two concrete examples

Create the runtime module in `src/erlab/interactive/` and keep any `.ui` file (if you
use Qt Designer) next to it. The rest of this page uses two real examples:

- a minimal tool that only implements the required
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` surface; and
- a fuller tool that also opts into manager metadata, copy-code support, and an
  ImageTool window kept with the tool in the manager.

### Minimal example: only the required {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` methods

If you only want to remember the minimum required pieces, this is it. The tool below is still
fully functional: it displays a scaled 2D array, saves and restores its state, and can
accept replacement data. It intentionally does **not** implement any of the optional
manager metadata or provenance hooks. It still implements
{meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>` because this
is the one refresh contract for every {class}`ToolWindow <erlab.interactive.utils.ToolWindow>`.

```python
from collections.abc import Mapping

import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtWidgets

import erlab


class MinimalScaleTool(erlab.interactive.utils.ToolWindow):
    tool_name = "scaletool"  # In practice, always set a stable user-facing tool name.

    class StateModel(pydantic.BaseModel):
        data_name: str
        scale: float = 1.0

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        super().__init__()

        # Store the main array and a stable variable name for saved state / reloads.
        self._data = self._coerce_data(data)
        self._data_name = data_name or (self._data.name or "data")

        # Build a normal central widget. ToolWindow wraps it in its own root widget.
        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        self.setCentralWidget(root)

        self.plot = pg.PlotWidget()
        self.image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.scale_spin = QtWidgets.QDoubleSpinBox()

        self.scale_spin.setRange(0.1, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.valueChanged.connect(self._refresh)

        self.plot.addItem(self.image)
        layout.addWidget(self.plot)
        layout.addWidget(self.scale_spin)

        # Paint the first frame after all widgets exist.
        self._refresh()
        self._reset_history_stack()

    def _coerce_data(self, data: xr.DataArray) -> xr.DataArray:
        # Minimal tools can share constructor and refresh validation in one helper.
        parsed = erlab.interactive.utils.parse_data(data)
        if parsed.ndim != 2:
            raise ValueError("`data` must be 2D")
        return parsed

    @property
    def tool_data(self) -> xr.DataArray:
        # ToolWindow stores this array separately from the UI state model.
        return self._data

    @property
    def tool_status(self) -> StateModel:
        # The getter must describe the current UI state.
        return self.StateModel(
            data_name=self._data_name,
            scale=float(self.scale_spin.value()),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        # The setter must fully restore the state captured by the getter.
        self._data_name = status.data_name
        self.scale_spin.setValue(status.scale)
        self._refresh()

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
        # Even a one-input tool receives the complete named input mapping.
        self._data = self._coerce_data(inputs["data"])
        with self._history_suppressed():
            self._refresh()
        self._reset_history_stack()
        return True

    def _display_data(self) -> xr.DataArray:
        return (self.tool_data * float(self.scale_spin.value())).rename(self._data_name)

    def _refresh(self) -> None:
        self.image.setDataArray(self._display_data())
        self._write_state()
```

That is the minimum {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` surface to
keep in your head:

- constructor with `data` as the primary positional input;
- nested {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>`;
- {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>`;
- {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` getter and setter;
  and
- {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>`.

Everything below is optional integration that you add when the tool needs it.

### Full example: a tool that works well inside the manager

The next example uses the same core
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` interface, but it also
implements the optional pieces that make a tool feel fully integrated with ERLabPy:
manager preview metadata, copy-code provenance, input validation, and an ImageTool
window kept with the tool row.

```python
import enum
import typing
from collections.abc import Mapping

import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab


class MyTool(erlab.interactive.utils.ToolWindow):
    tool_name = "mytool"

    class Output(enum.StrEnum):
        # Stable, serialized ids are what the manager stores in workspaces.
        FILTERED = "mytool.filtered"

    # Optional: describe the main "Copy Code" action declaratively.
    COPY_PROVENANCE: typing.ClassVar = (
        erlab.interactive.utils.ToolScriptProvenanceDefinition(
            start_label="Start from current mytool input data",
            label="Apply the current moving-average filter",
            expression_method="_filter_expression",
            assign="result",
        )
    )

    # Optional: declare an ImageTool window that appears under this tool in the manager.
    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        Output.FILTERED: erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_filtered_output",
            provenance=erlab.interactive.utils.ToolScriptProvenanceDefinition(
                start_label="Start from current mytool input data",
                label="Apply the current moving-average filter",
                expression_method="_filter_expression",
                assign="filtered",
            ),
        )
    }

    class StateModel(pydantic.BaseModel):
        data_name: str
        window: int = 1
        show_reference: bool = False

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        super().__init__()

        # Validate the input once up front and keep a stable variable name around.
        self._data = self._validate_data(data)
        self._data_name = data_name or (self._data.name or "data")
        self._filtered_itool: QtWidgets.QWidget | None = None

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        controls = QtWidgets.QHBoxLayout()
        self.setCentralWidget(root)

        # This example shows two image layers: the filtered output and the reference.
        self.plot = pg.PlotWidget()
        self.filtered_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.reference_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.window_spin = QtWidgets.QSpinBox()
        self.reference_check = QtWidgets.QCheckBox("Show reference")
        self.copy_btn = QtWidgets.QPushButton("Copy Code")
        self.open_filtered_btn = QtWidgets.QPushButton("Open filtered output")

        self.window_spin.setRange(1, 100)
        self.window_spin.setValue(1)
        self.window_spin.valueChanged.connect(self._controls_changed)
        self.reference_check.toggled.connect(self._controls_changed)
        # COPY_PROVENANCE only defines the generated code. A UI button still has to
        # connect to the built-in copy_code() slot explicitly.
        self.copy_btn.clicked.connect(self.copy_code)
        self.open_filtered_btn.clicked.connect(self.open_filtered)
        self.reference_image.setOpacity(0.35)

        self.plot.addItem(self.filtered_image)
        self.plot.addItem(self.reference_image)
        layout.addWidget(self.plot)
        controls.addWidget(self.window_spin)
        controls.addWidget(self.reference_check)
        controls.addWidget(self.copy_btn)
        controls.addWidget(self.open_filtered_btn)
        layout.addLayout(controls)

        self._refresh(notify=False)
        self._reset_history_stack()

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        # Optional: this is the thumbnail the manager shows for the tool.
        return self.filtered_image

    @property
    def info_text(self) -> str:
        # Optional: short HTML summary shown in the manager side panel.
        window = self._filter_window()
        shape = " x ".join(str(size) for size in self.tool_data.shape)
        return (
            f"<b>{self.tool_name}</b><br>"
            f"shape: {shape}<br>"
            f"window: {window}<br>"
            f"show reference: {self.reference_check.isChecked()}"
        )

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self._data_name,
            window=self._filter_window(),
            show_reference=self.reference_check.isChecked(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self._data_name = status.data_name
        with (
            QtCore.QSignalBlocker(self.window_spin),
            QtCore.QSignalBlocker(self.reference_check),
        ):
            self.window_spin.setValue(status.window)
            self.reference_check.setChecked(status.show_reference)
        self._refresh(notify=False)

    @staticmethod
    def _validate_data(data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(data)
        if data.ndim != 2:
            raise ValueError("`data` must be 2D")
        return data

    def validate_update_inputs(
        self, inputs: Mapping[str, xr.DataArray]
    ) -> Mapping[str, xr.DataArray]:
        validated = dict(super().validate_update_inputs(inputs))
        validated["data"] = self._validate_data(validated["data"])
        return validated

    def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
        # Inputs have already passed validate_update_inputs().
        status = self.tool_status
        with self._history_suppressed():
            self._data = inputs["data"]
            self.tool_status = status
            self._notify_data_changed()
        self._reset_history_stack()
        return True

    def _filter_window(self) -> int:
        return int(self.window_spin.value())

    def _filtered_output(self) -> xr.DataArray:
        # Optional output method used by IMAGE_TOOL_OUTPUTS.
        window = self._filter_window()
        filtered = self.tool_data.rolling(
            {dim: window for dim in self.tool_data.dims},
            center=True,
            min_periods=1,
        ).mean()
        return filtered.rename(f"{self._data_name}_filtered")

    def _filter_expression(
        self,
        *,
        primary_input: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        # Optional provenance method: return the final expression only.
        del data
        input_expr = primary_input or "data"
        window = self._filter_window()
        rolling_kwargs = ", ".join(f"{dim}={window}" for dim in self.tool_data.dims)
        return (
            f"{input_expr}.rolling({rolling_kwargs}, center=True, min_periods=1).mean()"
        )

    @QtCore.Slot()
    def open_filtered(self) -> None:
        # Declaring IMAGE_TOOL_OUTPUTS is not enough by itself. The tool still needs
        # an action that opens the ImageTool through _launch_output_imagetool().
        tool = self._launch_output_imagetool(
            self._filtered_output(),
            output_id=self.Output.FILTERED,
        )
        if tool is not None:
            self._filtered_itool = tool

    def _controls_changed(self, _value: int | bool) -> None:
        self._refresh()
        self._write_state()

    def _refresh(self, *, notify: bool = True) -> None:
        # Keep the on-screen view and any manager-facing outputs in sync.
        self.filtered_image.setDataArray(self._filtered_output())
        self.reference_image.setDataArray(self.tool_data, update_labels=False)
        self.reference_image.setVisible(self.reference_check.isChecked())
        if notify:
            self._notify_data_changed()
```

Some implementation details matter:

- Call `super().__init__()` before creating your UI.
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` installs the manager status
  banner and keyboard shortcuts.
- Always use
  {meth}`self.setCentralWidget() <erlab.interactive.utils.ToolWindow.setCentralWidget>`,
  not
  {meth}`QtWidgets.QMainWindow.setCentralWidget() <PySide6.QtWidgets.QMainWindow.setCentralWidget>`.
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` wraps the actual content
  widget so it can show update status above it.
- Keep {attr}`StateModel <erlab.interactive.utils.ToolWindow.StateModel>` focused on UI
  state. The main data already comes from
  {attr}`tool_data <erlab.interactive.utils.ToolWindow.tool_data>` and is stored
  separately in workspace files. If you need to persist expensive calculated arrays,
  use the explicit persistence hooks instead of
  {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` so ordinary
  history snapshots stay cheap.
- For a tool with several canonical inputs, override
  {meth}`_persistence_data_items() <erlab.interactive.utils.ToolWindow._persistence_data_items>`
  and store each non-primary input under its exact input name. The base class stores
  the primary input as `<saved-tool-data>`. Restore validates and applies the complete
  mapping once. Keep auxiliary arrays in
  {meth}`_restore_persistence_data_items() <erlab.interactive.utils.ToolWindow._restore_persistence_data_items>`.
- Keep workspace restore cheap. If a saved tool has optional cached results, preview
  images, deserialized fit objects, or rendered figures, validate the saved state
  eagerly but defer that derived work through
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` restore hooks. Hidden tools
  in a manager workspace should not deserialize, recompute, or render data until the
  user shows the tool or asks for its output.
- Record undo/redo checkpoints after user-visible state changes by calling
  {meth}`_write_state() <erlab.interactive.utils.ToolWindow._write_state>`. Call
  {meth}`_reset_history_stack() <erlab.interactive.utils.ToolWindow._reset_history_stack>`
  after construction, file restore, duplication, or source-data replacement so a
  restored or refreshed tool starts from the current state with no previous history.
  Use
  {meth}`_replace_last_state() <erlab.interactive.utils.ToolWindow._replace_last_state>`
  when an asynchronous result updates the current step instead of creating a new undo
  step.
- Make the {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` getter
  and setter fully describe and restore the current UI state. A restored tool should
  look the same as one configured interactively.
- Keep provenance and output declarations declarative. Prefer method names in
  {class}`ToolScriptProvenanceDefinition <erlab.interactive.utils.ToolScriptProvenanceDefinition>`
  and
  {class}`ToolImageOutputDefinition <erlab.interactive.utils.ToolImageOutputDefinition>`
  over inline lambdas so the class body remains readable and testable.
- If you want a visible "Copy Code" button, create that button in the UI and connect
  it to {meth}`self.copy_code() <erlab.interactive.utils.ToolWindow.copy_code>`.
  Declaring {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>`
  only tells {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` how to generate
  the code when that slot is called.
- Keep provenance methods on the shared
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` calling convention:
  `(*, primary_input: str | None = None, data: xr.DataArray | None = None)`. Most
  single-step methods should return only the unassigned final expression. Let
  {attr}`ToolScriptProvenanceDefinition.assign <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign>`
  or
  {attr}`assign_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign_method>`
  define the final variable name. Use
  {attr}`prelude_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.prelude_method>`
  only when the generated code needs setup statements before that final expression.
  Output-specific helpers can still inspect `data` when the generated code depends on
  the current output array.
- Prefer
  {attr}`expression_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.expression_method>`
  for ordinary single-step generated code. Use
  {attr}`operations_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.operations_method>`
  only when the generated code needs multiple labeled steps. Use
  {attr}`seed_code <erlab.interactive.utils.ToolScriptProvenanceDefinition.seed_code>`,
  {attr}`seed_code_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.seed_code_method>`,
  or
  {attr}`active_name <erlab.interactive.utils.ToolScriptProvenanceDefinition.active_name>`
  only when the simpler expression-plus-assignment path cannot describe the tool.
- If those multiple steps are one user action, keep the steps as primitive
  {class}`ToolProvenanceOperation <erlab.interactive.imagetool.provenance.ToolProvenanceOperation>`
  instances and add operation-group metadata instead of replacing them with a single
  raw script step. Use
  {func}`stamp_operation_group() <erlab.interactive.imagetool.provenance.stamp_operation_group>`
  with a stable `kind` string. It attaches the contiguous `index` and `size` markers
  and optional `focus` values for the dialog control that should receive focus when
  that row is edited. The grouped operations should still replay and generate readable
  public-API code one primitive step at a time.
- If generated code should be unavailable for the current state, return `None` from a
  dynamic provenance method such as
  {attr}`label_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.label_method>`,
  {attr}`assign_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign_method>`,
  or
  {attr}`prelude_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.prelude_method>`
  rather than returning partial code.
- If `_refresh()` changes manager-visible data, previews, or ImageTool windows opened
  from the tool, call
  {meth}`_notify_data_changed() <erlab.interactive.utils.ToolWindow._notify_data_changed>`
  from that path rather than emitting raw signals manually.
- If you want an ImageTool window to appear as a child row of the tool in the manager, do
  both pieces: declare it in
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` and
  open it through
  {meth}`_launch_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_output_imagetool>`
  with `output_id`. Declaring the output alone does not create any user-facing action.

{class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` in
{mod}`erlab.interactive.derivative` is a good synchronous example.
{attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` captures the
preprocessing controls, and
{meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>` swaps in the
new named input while preserving the current settings.

## Defer expensive restored work

Direct {meth}`ToolWindow.from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>`
calls restore tools eagerly, which keeps scripts and standalone file restores simple.
The ImageTool manager uses the same restore path with an internal deferred mode so
loading a workspace can finish before hidden tools rebuild optional cached state.

For new tools, split restore work into two categories:

- Required work: validation, source-reference resolution, UI state restoration, and
  anything needed for the tool to be structurally valid. Run this eagerly.
- Optional work: derived objects, rendered views, preview data, or result arrays that
  are expensive to materialize and can be recreated from saved state. Use the restore
  framework for this work. Existing examples include deserializing saved fit results,
  building a figure window, and preparing reduced preview data.

The normal pattern is:

```python
def _restore_persistence_payload(self, ds: xr.Dataset) -> None:
    self._serialized_result = np.array(ds["result_blob"].values, copy=True)
    self._run_or_defer_restore_work(
        self._restore_result_cache,
        run_on_show=True,
    )


def _restore_result_cache(self) -> None:
    self._result = deserialize_result(self._serialized_result)
    self._serialized_result = None
    self._notify_data_changed()
```

Use `run_on_show=True` when hidden tools can wait until the user shows them.
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` also flushes deferred work
before saving, copying generated code, returning declared ImageTool outputs, and other
correctness boundaries that consume the derived state. If the tool closes before any
such boundary, {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` discards the
queued work without running it. Do not use deferred restore callbacks for required
teardown or cleanup.

Call
{meth}`_flush_restore_work() <erlab.interactive.utils.ToolWindow._flush_restore_work>`
from a subclass only when that subclass has a narrower data boundary. For example,
{attr}`DerivativeTool.result <erlab.interactive.derivative.DerivativeTool.result>`
flushes the pending recomputation before returning the result array. Do not flush from
passive metadata paths merely to build manager labels or dependency status.

Call
{meth}`_discard_restore_work() <erlab.interactive.utils.ToolWindow._discard_restore_work>`
only when explicit work has superseded a queued restore callback. For example, if a
user-triggered preview update already rebuilt the same preview that was queued during
restore, discard the queued restore preview so it cannot run later and overwrite newer
state.

Use an explicit `key=` only when another path must address the deferred work by a
stable handle. The main use case is saving raw persisted payloads unchanged. For
example, a tool can keep a serialized result payload pending and have its save hook
flush other deferred work while skipping that payload, so a save immediately after
workspace load writes the original payload unchanged.

The restore hook should stay focused on derived work. Required validation and
source-reference resolution should run before any deferral, and passive manager
metadata paths should not flush restore work just to build labels or dependency status.

Tests for deferred restore should assert user-visible behavior instead of inspecting the
internal restore queue:

- direct {meth}`from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>` remains
  eager;
- manager-style deferred restore does not run the expensive callback during hidden load;
- showing the tool or requesting its output flushes the callback exactly once;
- closing a hidden tool discards callbacks whose results were never requested;
- saving before flush preserves any raw serialized payload that is meant to stay raw;
  and
- generated code, declared outputs, and provenance remain identical after flush.

## Add manager-facing metadata

The ImageTool manager can display a preview image and rich HTML summary for tools opened
from ImageTool.
These are optional, but tools feel much more integrated when they provide them.

The working `MyTool` reference above already implements both, so use it as the baseline
pattern for new synchronous tools.

Implement these properties when they make sense:

- {attr}`preview_imageitem <erlab.interactive.utils.ToolWindow.preview_imageitem>`:
  return the {class}`pyqtgraph.ImageItem` that should be rendered in the manager tree.
- {attr}`info_text <erlab.interactive.utils.ToolWindow.info_text>`: return a short HTML
  summary of the current tool state.

Whenever the preview or info text changes, emit
{attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>`. This is what
causes the manager to refresh its side panel and thumbnails.
{class}`KspaceToolGUI <erlab.interactive.kspace.KspaceToolGUI>` and
{class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` are good
references for this pattern.

If the tool can change its displayed data or any ImageTool window opened from the tool
without going through the built-in update flow, call
{meth}`self._notify_data_changed() <erlab.interactive.utils.ToolWindow._notify_data_changed>`.
That method emits both
{attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>` and
{attr}`sigDataChanged <erlab.interactive.utils.ToolWindow.sigDataChanged>`. These
signals let rows created by the tool become stale or update from the current tool state.
Emit {attr}`sigInfoChanged <erlab.interactive.utils.ToolWindow.sigInfoChanged>` directly
only for metadata-only changes.

## Support updates from ImageTool inputs

If a tool can be launched from ImageTool or opened from ImageTool in the manager, it
should usually be able to react when the ImageTool that opened it changes.

{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` gives you three hooks for this:

- {meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`:
  normalize or reject the complete named input mapping before it reaches the live UI.
- {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>`: apply the
  complete mapping without creating a brand-new window. Return `False` when the tool
  does not apply the input. Call `_defer_source_refresh()` before returning `False` when
  accepted asynchronous work will apply it later.
- {meth}`_cancel_background_work() <erlab.interactive.utils.ToolWindow._cancel_background_work>`:
  stop worker threads or queued tasks before mutating the UI, if your tool fits in the
  background.

There are three common update strategies in the current codebase:

1. In-place updates for simple tools.

   {class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` and
   {class}`KspaceToolGUI <erlab.interactive.kspace.KspaceToolGUI>` validate the
   `"data"` input, preserve
   {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`, replace their
   cached data, and recompute the plots.

2. Rebuild-and-restore updates for tools whose UI depends heavily on the input data.

   {class}`Fit1DTool <erlab.interactive._fit1d.Fit1DTool>` and
   {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>` snapshot
   {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>`, tear down the
   central widget, rebuild the UI, then restore the saved state. The base input
   transaction performs validation and background-task cancellation before it calls
   {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>`.

3. Deferred updates for tools that accept the new input before they can publish a fresh
   result.

   {class}`GoldTool <erlab.interactive.fermiedge.GoldTool>`,
   {class}`Fit1DTool <erlab.interactive._fit1d.Fit1DTool>`,
   {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>`, and
   {class}`ResolutionTool <erlab.interactive.fermiedge.ResolutionTool>` call
   {meth}`_defer_source_refresh() <erlab.interactive.utils.ToolWindow._defer_source_refresh>`
   after they start an asynchronous refit. They then return `False` to keep the input
   transaction pending. These tools publish the replacement input before the refit
   starts. They therefore call
   {meth}`finalize_source_refresh() <erlab.interactive.utils.ToolWindow.finalize_source_refresh>`
   when the refit succeeds, fails, is cancelled, or times out. Their fit-specific
   state still reports that the calculated result is stale or failed.

When your tool has worker threads, a typical pattern is:

```python
def _cancel_background_work(self, *, timeout_ms: int) -> bool:
    return self._threadpool.waitForDone(timeout_ms)


def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
    status = self.tool_status
    old_geom = self.saveGeometry()
    with self._history_suppressed():
        self._data = inputs["data"]
        self._rebuild_ui()
        self.tool_status = status
        self.restoreGeometry(old_geom)
        self._notify_data_changed()
    self._reset_history_stack()
    return True
```

If `update_inputs(...)` starts asynchronous follow-up work such as a refit, call
{meth}`_defer_source_refresh() <erlab.interactive.utils.ToolWindow._defer_source_refresh>`
and return `False`. Call
{meth}`finalize_source_refresh() <erlab.interactive.utils.ToolWindow.finalize_source_refresh>`
only after the new input or result is published. If the tool already published the new
input, also finalize the input transaction when a follow-up calculation fails or is
cancelled. Keep the calculation's own result state stale or failed. Call
{meth}`abort_source_refresh() <erlab.interactive.utils.ToolWindow.abort_source_refresh>`
only when the tool staged the input without publishing it. A bare `False` discards the
pending binding update. With automatic updates enabled, the framework coalesces source
changes that arrive during explicitly deferred work and applies the newest input after
the current work settles. With automatic updates disabled, it commits the completed
input and leaves the tool stale for the newer source. This prevents a new data array
from keeping the old input binding.

### Consume several Manager inputs

A {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` can depend on several
ImageTool or ToolWindow results. The Manager keeps these named dependencies as a graph
even though the tree shows the tool under only one row. Register all inputs together at
the launch site:

```python
from erlab.interactive.imagetool.provenance import ScriptInput


tool.set_script_inputs(
    (ScriptInput(name="data"), ScriptInput(name="right")),
    primary_input="data",
)
tool_uid = manager.add_childtool(
    tool,
    script_inputs={"data": data_target, "right": right_target},
    parent=data_target,
)
```

Each target is a Manager root index or node UID. The input names are durable identifiers
that the tool update and provenance code use. Their insertion order is also preserved.
The declaration fixes the names, roles, transforms, order, and primary input before the
Manager binds them to live targets. The default `data_role="displayed"` uses the values
currently displayed by an ImageTool, including an accepted filter. Use
`data_role="source"` only when the tool intentionally consumes the durable unfiltered
source array. The optional arguments have different purposes:

- `parent` selects the one input whose row contains the tool in the tree. It does not
  change the dependency graph.
- The
  {attr}`ToolWindow.primary_input <erlab.interactive.utils.ToolWindow.primary_input>`
  declaration selects the input used as the primary tool data. Structured provenance
  operations start from this input.

If `parent` is omitted, it defaults to the primary input. Removing a non-tree input
keeps the tool row and marks its source unavailable. Removing the tree parent removes
the nested tool with that branch. Workspace save, load, import, and node-UID rebasing
preserve all named bindings.

Implement one update transaction for the complete input mapping:

```python
from collections.abc import Mapping

import xarray as xr


def validate_update_inputs(
    self, inputs: Mapping[str, xr.DataArray]
) -> Mapping[str, xr.DataArray]:
    validated = dict(super().validate_update_inputs(inputs))
    left, right = xr.align(validated["data"], validated["right"], join="exact")
    validated["data"] = left
    validated["right"] = right
    return validated


def update_inputs(self, inputs: Mapping[str, xr.DataArray]) -> bool:
    self._data = inputs["data"]
    self._right = inputs["right"]
    self._refresh()
    return True
```

The framework checks that the names match the fixed bindings. It resolves all inputs,
calls
{meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`
once, and then calls
{meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>` once. The
tool does not observe a partly refreshed input set. For asynchronous work, call
{meth}`_defer_source_refresh() <erlab.interactive.utils.ToolWindow._defer_source_refresh>`,
return `False`, and then call
{meth}`finalize_source_refresh() <erlab.interactive.utils.ToolWindow.finalize_source_refresh>`
or {meth}`abort_source_refresh() <erlab.interactive.utils.ToolWindow.abort_source_refresh>`
when the work stops.

The Manager owns the complete input lifecycle. It resolves live nodes before it uses
recorded provenance, requests trust only for a recorded fallback, and propagates stale
or unavailable state through all active dependency edges. Workspace import remaps saved
node UIDs while it preserves saved revision tokens. Subtree duplication preserves the
initial revision tokens of the copied nodes and remaps binding UIDs. A
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` does not fetch Manager nodes or
select a replay policy. It only validates and applies the complete mapping that the
Manager supplies.

`script_inputs` and `primary_input` are the canonical input state on every refreshable
Manager child {class}`ToolWindow <erlab.interactive.utils.ToolWindow>`. Input names,
roles, transforms, order, and the primary input stay fixed after the first binding. The
Manager can refresh node UIDs, snapshot tokens, labels, and fallback provenance. Use
`add_childtool(..., script_inputs=...)` so target validation, dependency registration,
persistence, and refresh callbacks stay consistent. Figure Composer is a separate
Manager collection. It owns a dynamic `FigureSourceState` map instead of fixed child
tool inputs.

If the tool is launched from an ImageTool selection, the launch site should also record
which ImageTool data and selection opened it:

- Use
  {meth}`ItoolPlotItem.make_tool_source_spec() <erlab.interactive.imagetool.plot_items.ItoolPlotItem.make_tool_source_spec>`
  when the tool is created from the active cursor or cropped selection.
- Use {func}`full_data() <erlab.interactive.imagetool.provenance.full_data>` when the
  whole current array should be used again during an update.
- Store that live transform on the matching
  {attr}`ScriptInput.source_spec <erlab.interactive.imagetool.provenance.ScriptInput.source_spec>`.
  Store the already-transformed result's complete replay source on
  {attr}`ScriptInput.provenance_spec <erlab.interactive.imagetool.provenance.ScriptInput.provenance_spec>`.
  Replay uses this fallback as-is and does not apply the live transform again. The
  Manager updates the live node UID and snapshot token without changing the fixed name,
  role, or transform.
- To describe a tool's data transformations, import concrete operation models from
  ``erlab.interactive.imagetool._provenance._operations`` when a tool needs to write
  or modify the saved operation list explicitly. Pass those operation instances to
  source constructors such as
  {func}`selection() <erlab.interactive.imagetool.provenance.selection>` or
  {func}`full_data() <erlab.interactive.imagetool.provenance.full_data>`. The concrete operation
  catalog is still an internal, evolving interface; tool authors who depend on it are
  responsible for updating their integrations when those models change.
- If none of the existing models represents the transformation, implement a
  {class}`ToolProvenanceOperation <erlab.interactive.imagetool.provenance.ToolProvenanceOperation>`
  subclass. Give it
  a unique literal ``op`` value and Pydantic fields for every argument needed to repeat
  the operation. Implement
  {meth}`apply() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.apply>`,
  {meth}`derivation_label() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.derivation_label>`,
  and either
  {meth}`expression_code() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.expression_code>`
  for expression-based APIs or
  {meth}`statement_code() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.statement_code>`
  for a mutating API. Generated code must use public APIs and caller-provided variable
  names. Test both
  {meth}`apply() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.apply>`
  and the executed generated code against the same expected {class}`xarray.DataArray`.
- When a tool or dialog emits a sequence of primitive operations that should be edited
  as one unit, stamp the complete sequence as an operation group before returning it.
  The canonical order should be the order needed for clean generated code and replay.
  A transform dialog that edits such a group should set
  {attr}`operation_group_kind <erlab.interactive.imagetool.dialogs.DataTransformDialog.operation_group_kind>`,
  validate the selected contiguous group in
  {meth}`operation_group_for_edit() <erlab.interactive.imagetool.dialogs.DataTransformDialog.operation_group_for_edit>`,
  restore all controls from
  {meth}`restore_transform_operations() <erlab.interactive.imagetool.dialogs.DataTransformDialog.restore_transform_operations>`,
  and optionally use
  {meth}`focus_operation_group_control() <erlab.interactive.imagetool.dialogs.DataTransformDialog.focus_operation_group_control>`
  to focus the widget associated with the selected row. Rows copied without the full
  group must stay structured and replayable, but should not remain group-editable. When
  full groups are pasted, refresh their group identities before appending them to the
  destination provenance so adjacent copies remain separate editable groups.
- Each provenance operation type must have one editor registration. Do not list the
  same operation type in
  {attr}`operation_types <erlab.interactive.imagetool.dialogs.KspaceConversionDialog.operation_types>`
  for both a standalone dialog and a grouped dialog. A grouped dialog can include
  primitive operations owned by another editor without declaring those types in
  {attr}`operation_types <erlab.interactive.imagetool.dialogs.KspaceConversionDialog.operation_types>`;
  the group remains editable from its other registered operation rows. List these
  additional emitted operations in
  {attr}`batch_operation_types <erlab.interactive.imagetool.dialogs.KspaceConversionDialog.batch_operation_types>`
  when the grouped dialog supports batch use. If a batch group intentionally replaces
  existing coordinates, list those assignment types in
  {attr}`batch_coordinate_replacement_types <erlab.interactive.imagetool.dialogs.KspaceConversionDialog.batch_coordinate_replacement_types>`.
  Run
  `test_manager_provenance_operation_editor_contract_is_valid` after you change an
  editor registration.
- To make a transform or filter dialog reusable for manager step edits, keep the widget
  state restoration separate from the normal apply path. Transform dialogs should
  implement
  {meth}`restore_transform_operation() <erlab.interactive.imagetool.dialogs.DataTransformDialog.restore_transform_operation>`
  or
  {meth}`restore_transform_operations() <erlab.interactive.imagetool.dialogs.DataTransformDialog.restore_transform_operations>`
  and return current edits from
  {meth}`source_operations() <erlab.interactive.imagetool.dialogs.DataTransformDialog.source_operations>`.
  Filter dialogs should implement
  {meth}`restore_filter_operation() <erlab.interactive.imagetool.dialogs.DataFilterDialog.restore_filter_operation>`
  and
  {meth}`filter_operation() <erlab.interactive.imagetool.dialogs.DataFilterDialog.filter_operation>`.
  The manager opens the dialog on data replayed up to the edited step and reads the
  corresponding operation methods after the user accepts.
- When implementing a custom
  {meth}`ToolProvenanceOperation.derivation_entry() <erlab.interactive.imagetool.provenance.ToolProvenanceOperation.derivation_entry>`,
  return a {class}`DerivationEntry <erlab.interactive.imagetool.provenance.DerivationEntry>`
  for steps that should appear in the manager derivation list or
  copied code. Return ``None`` only for operations that must still run during an update
  but should stay hidden from the steps list and generated code, such as an internal
  bookkeeping rename. If the step should remain visible but code generation should
  stop, return
  {class}`DerivationEntry(..., code=None) <erlab.interactive.imagetool.provenance.DerivationEntry>`
  instead.
- Ensure the launch path declares `script_inputs` and `primary_input`. A standalone
  ImageTool installs one resolver for the complete mapping. The Manager registers the
  same bindings as graph dependencies. The
  {class}`ToolWindow <erlab.interactive.utils.ToolWindow>` does not fetch parent data.

If the tool offers "Copy Code" or otherwise generates code from its current input, also
implement provenance for that code path:

- Implement
  {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>` with a
  {class}`ToolScriptProvenanceDefinition <erlab.interactive.utils.ToolScriptProvenanceDefinition>`
  for the main copy-code action.
- Override
  {meth}`current_provenance_spec() <erlab.interactive.utils.ToolWindow.current_provenance_spec>`
  only when the declarative script metadata cannot describe the tool's generated code.
- Declare outputs in
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` when
  the tool exposes ImageTool windows as child rows whose generated code differs from
  the main tool action. The base
  {meth}`ToolWindow.output_imagetool_data() <erlab.interactive.utils.ToolWindow.output_imagetool_data>`
  and
  {meth}`ToolWindow.output_imagetool_provenance() <erlab.interactive.utils.ToolWindow.output_imagetool_provenance>`
  methods resolve those declared outputs for the manager. Authors should not override
  those methods for new outputs.
- Override
  {meth}`detached_output_imagetool_provenance() <erlab.interactive.utils.ToolWindow.detached_output_imagetool_provenance>`
  only when detached ImageTool launches should use different generated code from
  {meth}`current_provenance_spec() <erlab.interactive.utils.ToolWindow.current_provenance_spec>`.
  Call it before opening the window and pass its result explicitly:

  ```python
  def open_detached_output(self, output: xr.DataArray) -> None:
      self._launch_detached_output_imagetool(
          output,
          provenance_spec=self.detached_output_imagetool_provenance(output),
      )
  ```

  {meth}`_launch_detached_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool>`
  does not evaluate the hook itself. Return `None` or side-effect-free provenance
  instead of warning the user from inside the hook.

The full `MyTool` example above already shows the preferred pattern:

- {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>` describes
  the main copy-code path with a
  {class}`ToolScriptProvenanceDefinition <erlab.interactive.utils.ToolScriptProvenanceDefinition>`.
- `self.copy_btn.clicked.connect(self.copy_code)` wires a UI button to the built-in
  {meth}`copy_code() <erlab.interactive.utils.ToolWindow.copy_code>` slot.
- {class}`ToolScriptProvenanceDefinition <erlab.interactive.utils.ToolScriptProvenanceDefinition>`
  with
  {attr}`expression_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.expression_method>`
  and {attr}`assign <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign>`
  keeps the class declarative while the framework owns the final assignment target and
  active variable.
- {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>` with
  `Output.FILTERED` declares the filtered ImageTool window
  shown as a child row of the tool, with
  {attr}`data_method <erlab.interactive.utils.ToolImageOutputDefinition.data_method>` set
  to `"_filtered_output"` and a second provenance definition whose
  {attr}`assign <erlab.interactive.utils.ToolScriptProvenanceDefinition.assign>` target
  is `"filtered"`.
- The example's `open_filtered()` method uses
  {meth}`_launch_output_imagetool() <erlab.interactive.utils.ToolWindow._launch_output_imagetool>`
  with `output_id=self.Output.FILTERED` so the manager can persist and refresh that
  ImageTool window.

Use the current codebase as the source of truth for variants:

- {class}`DerivativeTool <erlab.interactive.derivative.DerivativeTool>` is the reference
  for
  {attr}`operations_method <erlab.interactive.utils.ToolScriptProvenanceDefinition.operations_method>`
  when generated code needs more than one operation. In this case, the tool does more
  than a single function call.
- {class}`KspaceTool <erlab.interactive.kspace.KspaceTool>`,
  {class}`GoldTool <erlab.interactive.fermiedge.GoldTool>`,
  {class}`MeshTool <erlab.interactive._mesh.MeshTool>`, and
  {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>` are good examples for ImageTool
  windows that appear as child rows of a tool in the manager.
- {class}`Fit1DTool <erlab.interactive._fit1d.Fit1DTool>` and
  {class}`Fit2DTool <erlab.interactive._fit2d.Fit2DTool>` are good main copy-code
  references.

The relevant examples live in
{class}`ItoolPlotItem <erlab.interactive.imagetool.plot_items.ItoolPlotItem>` and
{class}`ImageSlicerArea <erlab.interactive.imagetool.viewer.ImageSlicerArea>` as methods
named `open_in_<tool-name>`.

## Expose the tool cleanly

After the widget exists, add a public launcher function that users can call directly:

```python
import varname
import xarray as xr

import erlab


def mytool(
    data: xr.DataArray, data_name: str | None = None, *, execute: bool | None = None
) -> MyTool:
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=mytool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = MyTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()
    return win
```

This launcher is what should get the user-facing docstring. Treat it as part of the
real tool API, not as a thin convenience wrapper. Existing ERLabPy launchers typically
infer `data_name` here, then pass that stable name into the
{class}`ToolWindow <erlab.interactive.utils.ToolWindow>` instance so generated code and
saved state stay readable.

To make the tool discoverable across ERLabPy, update the relevant entry points:

- export it from `src/erlab/interactive/__init__.pyi`;
- add an IPython line magic in `src/erlab/interactive/_magic.py` if the tool is useful
  from notebooks;
- add ImageTool menu or context-menu actions if the tool operates on the current view or
  selection; and
- update the applicable GUI Reference and How-to pages so people can find it without
  reading the source.

If the tool should be available from an ImageTool in the manager, check both the plain
ImageTool launch path and the manager launch path. The manager flow is slightly
different because the tool row can be hidden, saved, restored, or rebound to watched
notebook data.

### Installed tool discovery

A workspace records the module and qualified class name for each saved tool. A fresh
Manager process accepts built-in tool classes and classes declared by installed
packages. It does not import an arbitrary module name from a workspace.

Add this entry point to the package that provides the tool:

```toml
[project.entry-points."erlab.interactive.tool_windows"]
my-tool = "my_package.tools:MyTool"
```

Include this entry point in the metadata of the installed package. Its value must match
the stored module and qualified class name. Manager reads installed entry-point
metadata before it imports the package. A class defined in a notebook or ordinary
script remains available after its module is imported in the current process, but a
fresh Manager process cannot discover it automatically.

## Test and document the contribution

Before opening a PR, make sure the new tool behaves like an ERLabPy tool, not just like
a local Qt app.

At minimum, add tests in `tests/interactive/test_<tool>.py` that cover:

- construction and basic interaction;
- {attr}`tool_status <erlab.interactive.utils.ToolWindow.tool_status>` serialization and
  restoration;
- {meth}`to_dataset() <erlab.interactive.utils.ToolWindow.to_dataset>` and
  {meth}`from_dataset() <erlab.interactive.utils.ToolWindow.from_dataset>` if the tool is
  savable, including any
  {meth}`_append_persistence_payload() <erlab.interactive.utils.ToolWindow._append_persistence_payload>`
  and
  {meth}`_restore_persistence_payload() <erlab.interactive.utils.ToolWindow._restore_persistence_payload>`
  roundtrip when the tool uses them;
- {meth}`validate_update_inputs() <erlab.interactive.utils.ToolWindow.validate_update_inputs>`
  and {meth}`update_inputs() <erlab.interactive.utils.ToolWindow.update_inputs>` as one
  atomic update, including the one-input case and {guilabel}`Stale` or
  {guilabel}`Unavailable` states after a source changes;
- multi-input bindings when applicable, including a removed non-tree input and
  workspace restore;
- deferred restore behavior for any expensive restored cache, preview, render, or
  result recomputation;
- dialog accept and cancel paths for any new dialogs, including {guilabel}`Save` and
  {guilabel}`Update Now` paths if the tool participates in automatic updates;
- manager launch paths, preferably by patching manager functions unless a live manager
  is required;
- generated code from
  {attr}`COPY_PROVENANCE <erlab.interactive.utils.ToolWindow.COPY_PROVENANCE>` and every
  {attr}`IMAGE_TOOL_OUTPUTS <erlab.interactive.utils.ToolWindow.IMAGE_TOOL_OUTPUTS>`
  definition that provides provenance: execute it in an explicit namespace and assert
  that the resulting object exactly matches the expected {class}`xarray.DataArray`; do
  not assert source formatting unless that exact formatting is the behavior under test;
- and grouped provenance behavior when applicable: serialization, full-group copy/paste
  preserving editability, partial copy/paste stripping group metadata while staying
  replayable, grouped edit replacement, grouped delete/revert behavior, and generated
  code execution for the grouped operation sequence.

If you add a new top-level test module, also update `scripts/_ci_test_groups.py` so the
CI shards still partition the suite correctly.

Document the new public entry point in two places:

- the launcher function and any public class docstrings; and
- the applicable GUI Reference page and any task-oriented How-to guide.

For GUI-facing contributions, include screenshots or a short recording in the PR, and
run the same checks expected for all contributions:

- `uv run ruff format .`
- `uv run ruff check --fix .`
- `uv run mypy src`
- `uv run pytest`
- `uv run python -m scripts.ci_test_groups --check-partition`

If you follow the patterns above, your tool will fit naturally into the existing
interactive ecosystem.

## Next steps

Once you have a working tool, you may want to contribute it to the repository. See the
{doc}`../contributing` for details on how to submit a pull request.
