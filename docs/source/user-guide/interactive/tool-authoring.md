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
  `from_file()`, using `tool_data`, `StateModel`, and `tool_status`;
- integration with the ImageTool manager, including tool naming, preview images, rich
  info text, and manager refresh notifications through `sigInfoChanged`;
- binding to ImageTool source data, including persisted source metadata, stale or
  unavailable status tracking, and the built-in source-update dialog; and
- the update hooks used by source-aware tools, such as `validate_update_data()`,
  `update_data()`, and `_cancel_background_work()`.

Use `ToolWindow` when your tool should do any of the following:

- accept an `xarray.DataArray` as its main input;
- round-trip through `to_dataset()` / `from_dataset()`;
- appear as a child tool inside the {ref}`ImageTool manager <imagetool-manager>` or
- refresh itself when the source ImageTool data changes.

`ToolWindow` assumes a few things about your implementation:

- The constructor accepts `data` as its single positional input. Additional options can
  be keyword arguments.
- The nested `StateModel` contains everything needed to restore the UI state.
- `tool_data` returns the main `DataArray`.
- `tool_status` serializes and reapplies the live widget state.

If your tool is a quick internal prototype or does not need save/restore support, a
plain Qt widget may be enough. For anything that should behave like `dtool`, `ftool`,
`goldtool`, or `ktool`, start from `ToolWindow`.

## Build a minimal `ToolWindow`

Create the runtime module in `src/erlab/interactive/` and keep any `.ui` file (if you
use Qt Designer) next to it. A minimal skeleton looks like this:

```python
import pydantic
import pyqtgraph as pg
import xarray as xr
from qtpy import QtWidgets

import erlab


class MyTool(erlab.interactive.utils.ToolWindow):
    tool_name = "mytool"

    class StateModel(pydantic.BaseModel):
        data_name: str
        sigma: float = 1.0
        show_reference: bool = False

    def __init__(self, data: xr.DataArray, *, data_name: str | None = None) -> None:
        super().__init__()

        self._data = self.validate_update_data(data)
        self._data_name = data_name or (self._data.name or "data")

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        self.setCentralWidget(root)

        self.plot = pg.PlotWidget()
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.reference_check = QtWidgets.QCheckBox("Show reference")

        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(1.0)
        self.sigma_spin.valueChanged.connect(self._refresh)
        self.reference_check.toggled.connect(self._refresh)

        layout.addWidget(self.plot)
        layout.addWidget(self.sigma_spin)
        layout.addWidget(self.reference_check)

        self._refresh()

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
        self._refresh()

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(new_data)
        if data.ndim != 2:
            raise ValueError("`data` must be 2D")
        return data

    def update_data(self, new_data: xr.DataArray) -> None:
        status = self.tool_status
        self._data = self.validate_update_data(new_data)
        self.tool_status = status
        self.sigInfoChanged.emit()

    def _refresh(self) -> None:
        ...
```

Some implementation details matter:

- Call `super().__init__()` before creating your UI. `ToolWindow` installs the manager
  status banner and keyboard shortcuts.
- Always use `self.setCentralWidget(...)`, not `QtWidgets.QMainWindow.setCentralWidget`.
  `ToolWindow` wraps the actual content widget so it can show source-update status above
  it.
- Keep `StateModel` focused on UI state. The main data already comes from `tool_data`
  and is stored separately when the tool is archived.
- Make the `tool_status` getter and setter a true round trip. A restored tool should
  look the same as one configured interactively.

`DerivativeTool` in `erlab.interactive.derivative` is a good synchronous example:
`tool_status` captures the preprocessing controls, and `update_data()` swaps in the new
array while preserving the current settings.

## Add manager-facing metadata

The ImageTool manager can display a preview image and rich HTML summary for child tools.
These are optional, but tools feel much more integrated when they provide them.

Implement these properties when they make sense:

- `preview_imageitem`: return the `pyqtgraph.ImageItem` that should be rendered in the
  manager tree.
- `info_text`: return a short HTML summary of the current tool state.

Whenever the preview or info text changes, emit `sigInfoChanged`. This is what causes
the manager to refresh its side panel and thumbnails. `KspaceToolGUI` and
`DerivativeTool` are good references for this pattern.

## Support source updates from ImageTool

If a tool can be launched from ImageTool or tracked by the manager, it should usually be
able to react when the parent data changes.

`ToolWindow` gives you three hooks for this:

- `validate_update_data(new_data)`: normalize or reject replacement data before it
  reaches the live UI.
- `update_data(new_data)`: apply the new data without creating a brand-new window.
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

3. Deferred updates for tools that cannot apply the data immediately.

   `ResolutionTool.update_data()` queues the request, aborts any in-flight fit, and
   returns `False` while work is still draining. Returning `False` keeps the tool marked
   as stale until the update actually completes.

When your tool has worker threads, a typical pattern is:

```python
def _cancel_background_work(self, *, timeout_ms: int) -> bool:
    return self._threadpool.waitForDone(timeout_ms)


def update_data(self, new_data: xr.DataArray) -> bool:
    status = self.tool_status
    old_geom = self.saveGeometry()

    def _apply_update(validated: xr.DataArray) -> None:
        self._data = validated
        self._rebuild_ui()
        self.tool_status = status
        self.restoreGeometry(old_geom)
        self.sigInfoChanged.emit()

    return self._perform_source_update(new_data, apply_update=_apply_update)
```

If the tool is launched from an ImageTool selection, the launch site should also bind
the tool back to its source data:

.. versionchanged:: 3.21.0

   Source-binding specs are now authored through
   ``erlab.interactive.imagetool.provenance``.

- Use `ItoolPlotItem.make_tool_source_spec(...)` when the tool is created from the
  active cursor or cropped selection.
- Use ``erlab.interactive.imagetool.provenance.full_data()`` when the whole current
  array is the logical source.
- Use the operation builders in ``erlab.interactive.imagetool.provenance`` such as
  ``selection(...)``, ``isel(...)``, ``sel(...)``, and ``average(...)`` when a tool
  needs to author or modify provenance explicitly.
- Ensure the caller sets `set_source_binding(...)`; the manager wrapper will provide
  `set_source_parent_fetcher(...)` when the tool is attached to a managed ImageTool.

The relevant examples live in `erlab.interactive.imagetool.plot_items.ItoolPlotItem` and
`erlab.interactive.imagetool.viewer.ImageSlicerArea` as methods named
`open_in_<tool-name>`.

## Expose the tool cleanly

After the widget exists, add a public launcher function that users can call directly:

```python
def mytool(
    data: xr.DataArray, data_name: str | None = None, *, execute: bool | None = None
) -> MyTool:
    with erlab.interactive.utils.setup_qapp(execute):
        win = MyTool(data, data_name=data_name)
        win.show()
        win.raise_()
        win.activateWindow()
    return win
```

This launcher is what should get the user-facing docstring.

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
- `tool_status` round-tripping;
- `to_dataset()` / `from_dataset()` if the tool is savable;
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
interactive ecosystem instead of becoming a one-off window that only works from a local
script.

## Next steps

Once you have a working tool, you may want to contribute it to the repository. See the
[contributing guide](../../contributing.md) for details on how to submit a pull request.
