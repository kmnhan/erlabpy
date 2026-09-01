(workflow-bridge)=
(interactive-tools)=

# Python and GUI workflows

Python scripts, notebooks, and GUI applications operate on labeled xarray objects. A
script stores analysis steps directly as code. ImageTool Manager records the source
and operation for supported GUI results and can generate Python code.

## Analysis interfaces

| Interface | Main strength | Suitable use |
| --- | --- | --- |
| Python script or notebook | Explicit and reusable analysis code | Complete analyses, automation, and repeated processing |
| ImageTool and specialized tools | Immediate visual feedback | Data inspection and parameter selection |
| ImageTool Manager | Source and result relationships, operation history, and saved workspaces | GUI analysis sessions and tracking derived results |
| Figure Composer | Editable Matplotlib recipes | Figure layout, reuse, and code generation |

The Python API provides all analysis features. GUI applications expose a smaller set
of analysis and plotting operations.

## Parallel notebook and Manager workflows

```{mermaid}
:alt: A notebook can synchronize a variable with ImageTool Manager, open data in Manager once, or run a standalone interactive tool until the tool closes.

flowchart TB
    notebook["Notebook<br/><small>DataArray variables · analysis code</small>"]
    watched["Watched variable<br/><small>%watch analysis_map</small>"]
    open["Open in Manager<br/><small>%itool -m · show_in_manager()</small>"]
    manager["ImageTool Manager<br/><small>ImageTools · managed tools · figures</small>"]
    result["Processed data<br/><small>Analysis steps recorded by Manager</small>"]
    code["Copied Python code"]
    tool["Standalone interactive tool<br/><small>dtool · ktool · goldtool · ftool<br/>Current notebook cell waits</small>"]
    continue["Continue notebook analysis"]

    notebook --> watched
    watched <-->|"Data updates"| manager
    notebook --> open --> manager
    manager -->|"Analyze data in the GUI"| result
    result -->|"Copy Full Code"| code --> continue
    notebook -->|"One-off analysis"| tool
    tool -->|"Close window"| continue

    class notebook input
    class watched,open,manager,tool focus
    class result,code,continue output
```

Use `%watch` when a notebook variable and a Manager row must stay synchronized. Use
`%itool -m` or {func}`show_in_manager
<erlab.interactive.imagetool.manager.show_in_manager>` when Manager only needs the
current data. When a supported GUI operation creates new data, Manager shows the result
below its source and records the operation in the {guilabel}`Provenance` tab. Use
{guilabel}`Copy Full Code` to copy the corresponding Python statements and continue the
analysis in the notebook.

Standalone tools such as {func}`erlab.interactive.dtool`,
{func}`erlab.interactive.ktool`, {func}`erlab.interactive.goldtool`, and
{func}`erlab.interactive.ftool` run in the notebook process. With the default Qt event
loop behavior, the current cell returns after the tool closes. This path is suitable
for one-off analysis that does not need Manager state.

## Data and code transfer

| Direction | Data or code |
| --- | --- |
| Notebook or script to ImageTool Manager | A shown or watched xarray object |
| GUI operation to Manager | A result `DataArray` grouped below the source that produced it |
| GUI operation to Python | Generated code that uses public Python APIs |
| Figure Composer to Python | Generated Matplotlib code |

ImageTool reads the same dimensions and coordinates that Python reads. Manager rows and
operation history do not change the dimensions, coordinates, or metadata of the
`DataArray`.

Interactive tools help users select data ranges and parameter values. Generated code
supports review, repeated analysis, and automation. Review generated code before
adding it to a script or notebook.

The table in {ref}`workflow-bridge-operations` lists GUI actions and their corresponding
Python operations. Use {doc}`../how-to/gui/python-integration` when you move data or
code between Python and the GUI. {doc}`../reference/gui/index` describes the
applications and controls. {doc}`workspaces-and-provenance` explains saved GUI state
and recorded operations.
