(workflow-bridge)=
(interactive-tools)=

# Python and GUI workflows

## Analysis workflows

ERLabPy supports three workflows. Two start from data and analysis code in a notebook.
The third starts and remains in ImageTool Manager.

```{mermaid}
:alt: Two panels show the notebook workflows and the Manager-only workflow.

flowchart LR
    subgraph notebook_workflows["Notebook workflows"]
        direction TB
        notebook["Notebook<br/><small>DataArray variables · analysis code</small>"]
        watched["Watch variable<br/><small>%watch analysis_map</small>"]
        show["Show current data<br/><small>%itool -m · show_in_manager()</small>"]
        manager["ImageTool Manager<br/><small>ImageTools · managed tools · figures</small>"]
        result["Derived result<br/><small>Provenance recorded</small>"]
        code["Copied Python code"]
        tool["Standalone tool<br/><small>dtool · ktool · goldtool · ftool</small>"]
        resume["Notebook analysis"]

        notebook --> watched
        watched <-->|"Synchronize variable and row"| manager
        notebook --> show --> manager
        manager -->|"Run managed operation"| result
        result -->|"Copy Full Code"| code --> resume
        notebook -->|"One-off analysis"| tool
        tool -->|"Close tool"| resume
    end

    subgraph manager_only_workflow["Manager-only workflow"]
        direction TB
        manager_only["ImageTool Manager<br/><small>Data · tools · figures</small>"]
        load["Load data<br/><small>Data Explorer · top-level row</small>"]
        analysis["Inspect and analyze<br/><small>ImageTool · ktool · dtool · ftool</small>"]
        managed_result["Derived results<br/><small>Nested rows · provenance</small>"]
        figure["Create figures<br/><small>Figure Composer</small>"]
        workspace["Save the session<br/><small>.itws workspace</small>"]

        manager_only --> load --> analysis --> managed_result
        managed_result --> figure
        managed_result --> workspace
    end

    notebook_workflows ~~~ manager_only_workflow

    class notebook,manager_only input
    class watched,show,manager,tool,load,analysis,figure focus
    class result,code,resume,managed_result,workspace output
```

### Notebook and Manager

Use this workflow when Python code remains the main analysis record and Manager
provides synchronized ImageTool windows, derived-result tracking, provenance, and
Figure Composer. Use `%watch` when a notebook variable and a Manager row must stay
synchronized. Use `%itool -m` or {func}`show_in_manager
<erlab.interactive.imagetool.manager.show_in_manager>` when Manager only needs the
current data.

When a supported GUI operation creates new data, Manager shows the result below its
source and records the operation in the {guilabel}`Provenance` tab. Use
{guilabel}`Copy Full Code` to copy the corresponding Python statements and continue the
analysis in the notebook.

### Notebook and standalone tools

Standalone tools such as {func}`erlab.interactive.dtool`,
{func}`erlab.interactive.ktool`, {func}`erlab.interactive.goldtool`, and
{func}`erlab.interactive.ftool` run in the notebook process. With the default Qt event
loop behavior, the current cell returns after the tool closes. This path is suitable
for one-off analysis that does not need Manager state.

### Manager-only workflow

Use this workflow when the analysis can remain in the GUI. Start ImageTool Manager and
use {ref}`imagetool-manager-data-explorer` to browse, preview, and load files. Loaded
data appears as a top-level row. Open it in ImageTool, apply supported operations or
open managed tools, and inspect derived results below their sources. Manager records
the corresponding operations as provenance.

Create managed figures with {ref}`figure-composer`. Save the session as an `.itws`
workspace to restore the Manager windows, derived relationships, provenance, and
figure recipes later.

## Data and code transfer

| Direction | Data or code |
| --- | --- |
| Files to ImageTool Manager | A loaded xarray object added as a top-level row |
| Notebook or script to ImageTool Manager | A shown or watched xarray object |
| GUI operation to Manager | A result `DataArray` grouped below the source that produced it |
| GUI operation to Python | Generated code that uses public Python APIs |
| Figure Composer to Python | Generated Matplotlib code |
| ImageTool Manager to workspace | Managed windows, state, row notes, derived relationships, and Figure Composer recipes |

ImageTool reads the same dimensions and coordinates that Python reads. Manager rows and
operation history do not change the dimensions, coordinates, or metadata of the
`DataArray`.

Interactive tools help users select data ranges and parameter values. Generated code
supports review, repeated analysis, and automation. Review generated code before
adding it to a script or notebook.

The table in {ref}`workflow-bridge-operations` lists GUI actions and their corresponding
Python operations. Use {doc}`../how-to/gui/python-integration` when you move data or
code between Python and the GUI. {doc}`../reference/gui/index` describes the
applications and controls. Use {doc}`../how-to/gui/derived-results-and-workspaces` to
update derived results, reuse recorded operations, or save a Manager workspace.
