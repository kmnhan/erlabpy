(interactive-tool-authoring)=

# Authoring interactive tools

This section is aimed at experienced users who are already comfortable with Qt,
`xarray`, and ERLabPy's analysis model, and want to contribute a new interactive tool to
{mod}`erlab.interactive`.

It focuses on ERLab-specific integration points: the `ToolWindow` class, manager
support, source updates, public launch paths, and the test/docs work expected in a
contribution. For general repository conventions, see the [contributing
guide](../../contributing.md).

## Start with the right shape

Most user-facing ERLabPy GUIs should inherit from
{class}`erlab.interactive.utils.ToolWindow` for it to correctly integrate with the
{ref}`ImageTool manager <imagetool-manager>`.

In practice, `ToolWindow` enables several things:

- save/restore support through `to_dataset()`, `from_dataset()`, `to_file()`, and
  `from_file()`, using `tool_data`, `StateModel`, `tool_status`, and optional
  save-only payload hooks for persisted data that should stay out of undo/redo history;
- integration with the ImageTool manager, including tool naming, preview images, rich
  info text, and manager refresh notifications through `sigInfoChanged`;
- binding to ImageTool source data, including persisted source metadata, stale or
  unavailable status tracking, and the built-in source-update dialog; and
- child ImageTool windows declared in `IMAGE_TOOL_OUTPUTS` that appear beneath the
  tool's node in the ImageTool manager and can be reopened, refreshed, and persisted;
  and
- the update hooks used by source-aware tools, such as `validate_update_data()`,
  `update_data()`, and `_cancel_background_work()`.

Use `ToolWindow` when your tool should do any of the following:

- accept an `xarray.DataArray` as its main input;
- serialize and restore itself through `to_dataset()` / `from_dataset()`;
- appear as a child tool inside the {ref}`ImageTool manager <imagetool-manager>` or
- refresh itself when the source ImageTool data changes.

`ToolWindow` assumes a few things about your implementation:

- The constructor accepts `data` as its single positional input. Additional options can
  be keyword arguments.
- The nested `StateModel` contains the lightweight UI state that participates in
  undo/redo history.
- `tool_data` returns the main `DataArray`.
- `tool_status` serializes and reapplies the live widget state.
- If the tool needs extra persisted state that should not participate in history,
  override `_append_persistence_payload()` and `_restore_persistence_payload()`.

As a practical authoring checklist:

- Required for the core `ToolWindow` interface:
  constructor with `data`, `StateModel`, `tool_data`, and `tool_status`.
- Required if the tool can be refreshed from an ImageTool source:
  `update_data()`. In practice, this is the normal baseline for repository tools, so
  the examples below implement it even in the minimal case. Return `False` when the
  new source data is accepted but the tool cannot publish a fresh result yet.
- Optional, but strongly recommended for user-facing tools:
  `tool_name` (the base class default is just `"tool"`).
- Optional for tools with expensive or bulky save-only state:
  `_append_persistence_payload()` and `_restore_persistence_payload()`.
- Optional manager / provenance integration:
  `validate_update_data()`, `_cancel_background_work()`, `preview_imageitem`,
  `info_text`, `COPY_PROVENANCE`, `IMAGE_TOOL_OUTPUTS`, and
  `detached_output_imagetool_provenance()`.

If your tool is a quick internal prototype or does not need save/restore support, a
plain Qt widget may be enough. For anything that should behave like `dtool`, `ftool`,
`goldtool`, or `ktool`, start from `ToolWindow`.

## Map capabilities to hooks

When you add a new tool, think in terms of user-visible capabilities first and then wire
up the corresponding `ToolWindow` surface:

- Save and restore the tool window:
  required; implement `StateModel`, `tool_status`, and `tool_data`. If save/load also
  needs expensive derived results, keep them out of `tool_status` and use
  `_append_persistence_payload()` / `_restore_persistence_payload()` instead.
- Show rich metadata in the ImageTool manager:
  optional; implement `info_text`, `preview_imageitem`, and emit `sigInfoChanged` when
  either changes.
- Refresh the tool from parent ImageTool data:
  `update_data()` is part of the minimal tool surface; `validate_update_data()` and
  `_cancel_background_work()` are optional additions when normalization or worker
  shutdown matter.
- Generate replayable code for the tool's main action:
  optional; usually set `COPY_PROVENANCE` to a `ToolScriptProvenanceDefinition`. Prefer
  `label + expression_method + assign` for the common single-step case, add
  `prelude_method` when the replay code needs setup statements before the final
  expression, and use `operations_method` only when the copied replay script truly
  needs multiple labeled operations.
- Expose refreshable child ImageTool windows beneath the tool's manager node:
  optional; declare a stable output id, preferably with `enum.StrEnum`, and add it to
  `IMAGE_TOOL_OUTPUTS` with a `ToolImageOutputDefinition(data_method=...,
  provenance=...)`. `data_method` should name a zero-argument instance method that
  returns the current output `DataArray`. The string output id is what the manager
  persists in saved workspaces, so use a tool-qualified name such as
  `"mytool.filtered"`.
- Open an ImageTool that is not a declared manager-tracked output binding:
  optional; call `_launch_detached_output_imagetool(...)`. In the manager, that opens
  a fresh independent top-level ImageTool window with no parent/source/output binding.
  Outside the manager, each call opens a new standalone ImageTool window. Override
  `detached_output_imagetool_provenance()` only when standalone launches should show
  different replay lineage from `current_provenance_spec()`, and keep that hook free
  of blocking side effects such as modal warnings.

The important distinction is that outputs declared in `IMAGE_TOOL_OUTPUTS` become child
ImageTool windows beneath the tool's node in the manager, keyed by a serialized
`output_id`. Any ImageTool opened without an `output_id` is not reproducible as one of
those child windows. Do not use `_launch_detached_output_imagetool(...)` as a
substitute for a real declared child ImageTool output.

A real example is `Fit2DTool`:

- `Fit2DTool.Output.PARAMETER_VALUES` and `Fit2DTool.Output.PARAMETER_STDERR` are
  declared in `IMAGE_TOOL_OUTPUTS`, so those parameter plots become child ImageTool
  windows beneath the fit tool's node, with persisted `output_id`s.
- `Fit2DTool._show_dataarray_in_itool()` also has a generic path for arbitrary
  `DataArray`s that are not declared outputs. In the manager, that path opens a fresh
  independent top-level ImageTool window each time. Outside the manager, it opens a
  fresh standalone ImageTool window each time.
- That generic path is intentionally not a nested child binding because the manager
  cannot recreate it from either `source_spec` or `output_id`.

## Build two concrete examples

Create the runtime module in `src/erlab/interactive/` and keep any `.ui` file (if you
use Qt Designer) next to it. The rest of this page uses two real examples:

- a minimal tool that only implements the required `ToolWindow` surface; and
- a fuller tool that also opts into manager metadata, replayable copy-code support, and
  a child ImageTool output shown beneath the tool in the manager.

### Minimal example: only the required `ToolWindow` methods

If you only want to remember the minimum required pieces, this is it. The tool below is still
fully functional: it displays a scaled 2D array, saves and restores its state, and can
accept replacement data. It intentionally does **not** implement any of the optional
manager niceties or provenance hooks, but it still includes `update_data()` because
that is the practical baseline for tools that may be launched from ImageTool.

```python
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

    def _coerce_data(self, data: xr.DataArray) -> xr.DataArray:
        # Minimal tools can validate inline instead of overriding validate_update_data().
        parsed = erlab.interactive.utils.parse_data(data)
        if parsed.ndim != 2:
            raise ValueError("`data` must be 2D")
        return parsed

    @property
    def tool_data(self) -> xr.DataArray:
        # ToolWindow archives this array separately from the UI state model.
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

    def update_data(self, new_data: xr.DataArray) -> bool:
        # This is the minimal refresh path: replace the data and repaint.
        self._data = self._coerce_data(new_data)
        self._refresh()
        return True

    def _display_data(self) -> xr.DataArray:
        return (self.tool_data * float(self.scale_spin.value())).rename(self._data_name)

    def _refresh(self) -> None:
        self.image.setDataArray(self._display_data())
```

That is the minimum `ToolWindow` surface to keep in your head:

- constructor with `data` as the single positional argument;
- nested `StateModel`;
- `tool_data`;
- `tool_status` getter and setter; and
- `update_data`.

Everything below is optional integration that you add when the tool needs it.

### Full example: a manager-aware tool that implements the optional hooks too

The next example uses the same core `ToolWindow` interface, but it also implements the
optional pieces that make a tool feel fully integrated with ERLabPy: manager preview
metadata, copy-code provenance, source validation, and a child ImageTool output.

```python
import enum
import typing

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

    # Optional: declare a child ImageTool window that appears under this tool in the manager.
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
        sigma: float = 1.0
        show_reference: bool = False

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        super().__init__()

        # Validate the input once up front and keep a stable variable name around.
        self._data = self.validate_update_data(data)
        self._data_name = data_name or (self._data.name or "data")
        self._filtered_itool: QtWidgets.QWidget | None = None

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        controls = QtWidgets.QHBoxLayout()
        self.setCentralWidget(root)

        # This example shows two image layers: the filtered output and the reference.
        self.plot = pg.PlotWidget()
        self.filtered_image = erlab.interactive.utils.xImageItem(axisOrder="row-major")
        self.reference_image = erlab.interactive.utils.xImageItem(
            axisOrder="row-major"
        )
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.reference_check = QtWidgets.QCheckBox("Show reference")
        self.copy_btn = QtWidgets.QPushButton("Copy Code")
        self.open_filtered_btn = QtWidgets.QPushButton("Open filtered output")

        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(1.0)
        self.sigma_spin.valueChanged.connect(self._refresh)
        self.reference_check.toggled.connect(self._refresh)
        # COPY_PROVENANCE only defines the replay code. A UI button still has to
        # connect to the built-in copy_code() slot explicitly.
        self.copy_btn.clicked.connect(self.copy_code)
        self.open_filtered_btn.clicked.connect(self.open_filtered)
        self.reference_image.setOpacity(0.35)

        self.plot.addItem(self.filtered_image)
        self.plot.addItem(self.reference_image)
        layout.addWidget(self.plot)
        controls.addWidget(self.sigma_spin)
        controls.addWidget(self.reference_check)
        controls.addWidget(self.copy_btn)
        controls.addWidget(self.open_filtered_btn)
        layout.addLayout(controls)

        self._refresh(notify=False)

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        # Optional: this is the thumbnail the manager shows for the tool.
        return self.filtered_image

    @property
    def info_text(self) -> str:
        # Optional: short HTML summary shown in the manager side panel.
        sigma = float(self.sigma_spin.value())
        shape = " x ".join(str(size) for size in self.tool_data.shape)
        return (
            f"<b>{self.tool_name}</b><br>"
            f"shape: {shape}<br>"
            f"window: {self._filter_window()}<br>"
            f"show reference: {self.reference_check.isChecked()}<br>"
            f"sigma spin value: {sigma:g}"
        )

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self._data_name,
            sigma=float(self.sigma_spin.value()),
            show_reference=self.reference_check.isChecked(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self._data_name = status.data_name
        self.sigma_spin.setValue(status.sigma)
        self.reference_check.setChecked(status.show_reference)
        self._refresh(notify=False)

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        # Optional but recommended: normalize / reject source updates in one place.
        data = erlab.interactive.utils.parse_data(new_data)
        if data.ndim != 2:
            raise ValueError("`data` must be 2D")
        return data

    def update_data(self, new_data: xr.DataArray) -> bool:
        # Preserve the existing UI state while swapping in replacement data.
        status = self.tool_status
        self._data = self.validate_update_data(new_data)
        self.tool_status = status
        self._notify_data_changed()
        return True

    def _filter_window(self) -> int:
        return max(1, int(round(self.sigma_spin.value())))

    def _filtered_output(self) -> xr.DataArray:
        # Optional output helper used by IMAGE_TOOL_OUTPUTS.
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
        input_name: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        # Optional provenance helper: return the final expression only.
        del data
        input_expr = input_name or "data"
        window = self._filter_window()
        rolling_kwargs = ", ".join(
            f"{dim}={window}" for dim in self.tool_data.dims
        )
        return (
            f"{input_expr}.rolling("
            f"{rolling_kwargs}, center=True, min_periods=1"
            ").mean()"
        )

    @QtCore.Slot()
    def open_filtered(self) -> None:
        # Declaring IMAGE_TOOL_OUTPUTS is not enough by itself. The tool still needs
        # an action that opens the child ImageTool through _launch_output_imagetool().
        tool = self._launch_output_imagetool(
            self._filtered_output(),
            output_id=self.Output.FILTERED,
        )
        if tool is not None:
            self._filtered_itool = tool

    def _refresh(self, *, notify: bool = True) -> None:
        # Keep the on-screen view and any manager-facing outputs in sync.
        self.filtered_image.setDataArray(self._filtered_output())
        self.reference_image.setDataArray(self.tool_data, update_labels=False)
        self.reference_image.setVisible(self.reference_check.isChecked())
        if notify:
            self._notify_data_changed()
```

Some implementation details matter:

- Call `super().__init__()` before creating your UI. `ToolWindow` installs the manager
  status banner and keyboard shortcuts.
- Always use `self.setCentralWidget(...)`, not `QtWidgets.QMainWindow.setCentralWidget`.
  `ToolWindow` wraps the actual content widget so it can show source-update status above
  it.
- Keep `StateModel` focused on UI state. The main data already comes from `tool_data`
  and is stored separately when the tool is archived. If you need to persist expensive
  derived results, use the explicit persistence hooks instead of `tool_status` so
  ordinary history snapshots stay cheap.
- Make the `tool_status` getter and setter fully describe and restore the current UI
  state. A restored tool should look the same as one configured interactively.
- Keep provenance and output declarations declarative. Prefer method names in
  `ToolScriptProvenanceDefinition` and `ToolImageOutputDefinition` over inline lambdas
  so the class body remains readable and testable.
- If you want a visible "Copy Code" button, create that button in the UI and connect
  it to `self.copy_code`. Declaring `COPY_PROVENANCE` only tells `ToolWindow` how to
  generate the replay code when that slot is called.
- Keep provenance helper methods on the shared `ToolWindow` calling convention:
  `(*, input_name: str | None = None, data: xr.DataArray | None = None)`. Most
  single-step helpers should return only the unassigned final expression. Let
  `ToolScriptProvenanceDefinition(assign=...)` or `assign_method=...` define the final
  variable name, and use `prelude_method` only when the replay code needs setup
  statements before that final expression. Output-specific helpers can still inspect
  `data` when the generated replay code depends on the current output array.
- Prefer `expression_method` for ordinary single-step replay code. Use
  `operations_method` only when the replay needs multiple labeled steps, and reach for
  `seed_code`, `seed_code_method`, or explicit `active_name` only for the rarer cases
  where the simpler expression-plus-assignment path cannot describe the tool cleanly.
- If replay code should be unavailable for the current state, return `None` from a
  dynamic provenance helper such as `label_method`, `assign_method`, or
  `prelude_method` rather than returning partial code.
- If `_refresh()` changes manager-visible data, previews, or child outputs, call
  `_notify_data_changed()` from that path rather than emitting raw signals manually.
- If you want a child ImageTool window to appear beneath the tool's node in the
  manager, do both pieces: declare it in `IMAGE_TOOL_OUTPUTS` and open it through
  `_launch_output_imagetool(..., output_id=...)`. Declaring the output alone does not
  create any user-facing action.

`DerivativeTool` in `erlab.interactive.derivative` is a good synchronous example:
`tool_status` captures the preprocessing controls, and `update_data()` swaps in the new
array while preserving the current settings.

## Add manager-facing metadata

The ImageTool manager can display a preview image and rich HTML summary for child tools.
These are optional, but tools feel much more integrated when they provide them.

The working `MyTool` reference above already implements both, so use it as the baseline
pattern for new synchronous tools.

Implement these properties when they make sense:

- `preview_imageitem`: return the `pyqtgraph.ImageItem` that should be rendered in the
  manager tree.
- `info_text`: return a short HTML summary of the current tool state.

Whenever the preview or info text changes, emit `sigInfoChanged`. This is what causes
the manager to refresh its side panel and thumbnails. `KspaceToolGUI` and
`DerivativeTool` are good references for this pattern.

If the tool can change its displayed data or any manager-visible ImageTool outputs
without going through the built-in source-refresh flow, call
`self._notify_data_changed()`. That helper emits both `sigInfoChanged` and
`sigDataChanged`, which is what lets managed descendants become stale or auto-refresh
from the current tool state. Emit `sigInfoChanged` directly only for metadata-only
changes.

## Support source updates from ImageTool

If a tool can be launched from ImageTool or tracked by the manager, it should usually be
able to react when the parent data changes.

`ToolWindow` gives you three hooks for this:

- `validate_update_data(new_data)`: normalize or reject replacement data before it
  reaches the live UI.
- `update_data(new_data)`: apply the new data without creating a brand-new window.
  Return `False` when the input was accepted but the tool must stay stale until a
  deferred recomputation or result publication finishes.
- `_cancel_background_work(timeout_ms=...)`: stop worker threads or queued tasks before
  mutating the UI, if your tool fits in the background.

There are three common update strategies in the current codebase:

1. In-place updates for simple tools.

   `DerivativeTool` and `KspaceToolGUI` validate the new array, preserve `tool_status`,
   replace their cached data, and recompute the plots.

2. Rebuild-and-restore updates for tools whose UI depends heavily on the input data.

   `Fit1DTool` and `Fit2DTool` snapshot `tool_status`, tear down the central widget,
   rebuild the UI, then restore the saved state. In that case, prefer
   `self._perform_source_update(...)` so validation and background-task cancellation
   stay in one place.

3. Deferred updates for tools that accept the new input before they can publish a fresh
   result.

   `GoldTool.update_data()` returns `False` while a queued source update or refit is
   still pending, and `Fit1DTool`, `Fit2DTool`, and `ResolutionTool` return `False`
   when they have accepted new source data but must finish an asynchronous refit before
   their current outputs are fresh again. Returning `False` keeps the tool marked as
   stale until that follow-up work finishes and the tool calls
   `finalize_source_refresh()`.

When your tool has worker threads, a typical pattern is:

```python
def _cancel_background_work(self, *, timeout_ms: int) -> bool:
    return self._threadpool.waitForDone(timeout_ms)


def update_data(self, new_data: xr.DataArray) -> bool:
    status = self.tool_status
    old_geom = self.saveGeometry()

    def _apply_update(validated: xr.DataArray) -> bool:
        self._data = validated
        self._rebuild_ui()
        self.tool_status = status
        self.restoreGeometry(old_geom)
        self._notify_data_changed()
        return True

    return self._perform_source_update(new_data, apply_update=_apply_update)
```

If `_apply_update(...)` starts asynchronous follow-up work such as a refit, return
`False` instead and call `finalize_source_refresh()` only after the new result has been
published. This is what prevents manager-tracked descendants from refreshing against
stale derived outputs.

If the tool is launched from an ImageTool selection, the launch site should also bind
the tool back to its source data:

- Use `ItoolPlotItem.make_tool_source_spec(...)` when the tool is created from the
  active cursor or cropped selection.
- Use ``erlab.interactive.imagetool.provenance.full_data()`` when the whole current
  array is the logical source.
- Use the operation models in ``erlab.interactive.imagetool.provenance`` such as
  ``QSelOperation(...)``, ``IselOperation(...)``, ``SelOperation(...)``,
  ``AverageOperation(...)``, and ``TransposeOperation(...)`` when a tool needs to
  author or modify provenance explicitly. Pass those operation instances to
  ``selection(...)`` or ``full_data(...)``.
- When implementing a custom ``ToolProvenanceOperation.derivation_entry()``, return a
  ``DerivationEntry`` for steps that should appear in the manager derivation list or
  copied provenance code. Return ``None`` only for operations that must still replay at
  runtime but should stay hidden from the derivation UI and generated code, such as an
  internal bookkeeping rename. If the step should remain visible but code generation
  should stop, return ``DerivationEntry(..., code=None)`` instead.
- Ensure the caller sets `set_source_binding(...)`; the manager wrapper will provide
  `set_source_parent_fetcher(...)` and `set_input_provenance_parent_fetcher(...)` when
  the tool is attached to a managed ImageTool.

If the tool offers "Copy Code" or otherwise generates replayable code from its current
input, also implement provenance for that code path:

- Implement `COPY_PROVENANCE` with a `ToolScriptProvenanceDefinition` for the main
  copy-code action.
- Override `current_provenance_spec()` only when the declarative script metadata cannot
  describe the tool's replay code.
- Declare outputs in `IMAGE_TOOL_OUTPUTS` when the tool exposes child ImageTool windows
  beneath its manager node whose replay code differs from the main tool action. The
  base `ToolWindow.output_imagetool_data()` and
  `ToolWindow.output_imagetool_provenance()` methods resolve those declared outputs for
  the manager, so authors should not override those methods for new outputs.
- Override `detached_output_imagetool_provenance()` only when non-bound standalone
  ImageTool launches should use different replay lineage from `current_provenance_spec()`.
  This provenance is evaluated while opening the new window, so return `None` or
  side-effect-free provenance instead of warning the user from inside this hook.

The full `MyTool` example above already shows the preferred pattern:

- `COPY_PROVENANCE` describes the main copy-code path with a
  `ToolScriptProvenanceDefinition`.
- `self.copy_btn.clicked.connect(self.copy_code)` wires a UI button to the built-in
  copy-code slot.
- `ToolScriptProvenanceDefinition(expression_method=..., assign=...)` keeps the class
  declarative while the framework owns the final assignment target and active variable.
- `IMAGE_TOOL_OUTPUTS[Output.FILTERED]` declares the filtered child ImageTool output
  shown beneath the tool's manager node, with `data_method="_filtered_output"` and a
  second provenance definition whose `assign` target is `"filtered"`.
- `open_filtered()` uses `_launch_output_imagetool(..., output_id=self.Output.FILTERED)`
  so the manager can persist and refresh that child output.

Use the current codebase as the source of truth for variants:

- `DerivativeTool` is the reference for `operations_method` when a replay script needs
  more than one operation, i.e., the tool does more than a single function call.
- `KspaceTool`, `GoldTool`, `MeshTool`, and `Fit2DTool` are good manager-output
  references.
- `Fit1DTool` and `Fit2DTool` are good main copy-code references.

The relevant examples live in `erlab.interactive.imagetool.plot_items.ItoolPlotItem` and
`erlab.interactive.imagetool.viewer.ImageSlicerArea` as methods named
`open_in_<tool-name>`.

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
real tool API, not as a thin convenience wrapper: built-in tools typically infer
`data_name` here, then pass that stable name into the `ToolWindow` instance so replay
code and saved state stay readable.

To make the tool discoverable across ERLabPy, update the relevant entry points:

- export it from `src/erlab/interactive/__init__.pyi`;
- add an IPython line magic in `src/erlab/interactive/_magic.py` if the tool is useful
  from notebooks;
- add ImageTool menu or context-menu actions if the tool operates on the current view or
  selection; and
- update the user guide so people can find it without reading the source.

If the tool should be available from a managed ImageTool, check both the unmanaged and
managed launch paths. Manager-aware flows are slightly different because the child tool
can be hidden, archived, restored, or rebound to watched notebook data.

## Test and document the contribution

Before opening a PR, make sure the new tool behaves like an ERLabPy tool, not just like
a local Qt app.

At minimum, add tests in `tests/interactive/test_<tool>.py` that cover:

- construction and basic interaction;
- `tool_status` serialization and restoration;
- `to_dataset()` / `from_dataset()` if the tool is savable, including any
  `_append_persistence_payload()` / `_restore_persistence_payload()` roundtrip when the
  tool uses them;
- `validate_update_data()` and `update_data()` branches, including stale or unavailable
  source cases when relevant;
- dialog accept and cancel paths for any new dialogs; and
- manager-aware dispatch paths, preferably by patching manager helpers unless a live
  manager is required.

If you add a new top-level test module, also update `scripts/_ci_test_groups.py` so the
CI shards still partition the suite correctly.

Document the new public entry point in two places:

- the launcher function and any public class docstrings; and
- the user guide page where users would naturally look for the tool.

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
[contributing guide](../../contributing.md) for details on how to submit a pull request.
