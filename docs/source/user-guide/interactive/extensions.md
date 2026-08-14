(imagetool-manager-extensions)=

# Extend ImageTool Manager

ImageTool Manager extensions let you add lab-specific analysis routines and data
loaders without changing ERLabPy. A basic extension is one Python file with one or
more decorated functions. You do not need to use Qt, manager objects, provenance
classes, or workspace serialization APIs.

Decorated functions remain normal Python functions. You can call them directly in a
notebook, register them with ImageTool Manager, or load the file with the public
{mod}`erlab.extensions` API.

:::{warning}

An extension runs Python code in the ImageTool Manager process. It has the same access
to your files and system as ERLabPy. Review the source and add extensions only from
sources that you trust.

:::

## Choose an extension format

Use a script for most lab routines. Use a package when your extension already belongs
to an installable Python project.

| Format | Use it when | How it is shared |
| --- | --- | --- |
| Python script | You want the shortest authoring and installation path. | Share the `.py` file through version control, a shared directory, email, or a workspace. Each user registers the file. |
| Python package | You maintain dependencies and releases with standard Python packaging tools. | Install the package in the active Python environment, or include it when you build the standalone application. |

ImageTool Manager does not maintain a user-visible revision history for a script. Your
source repository is responsible for version history. The manager records a source
hash internally so that it can detect changed files and preserve the exact source used
by a workspace.

## Write an analysis routine

The following file defines a Gaussian convolution routine. Save it as
`gaussian_tools.py`.

```python
from typing import Literal

import xarray as xr

import erlab.analysis as era
from erlab.extensions import routine


@routine(
    name="Gaussian convolution",
    category="My Lab",
    summary="Apply a coordinate-aware Gaussian convolution.",
)
def gaussian_convolution(
    data: xr.DataArray,
    sigma: float = 0.01,
    mode: Literal["nearest", "reflect", "constant"] = "nearest",
) -> xr.DataArray:
    return era.image.gaussian_filter(data, sigma=sigma, mode=mode)
```

The `sigma` value uses the coordinates of each data dimension, not pixel units. The
function can be called normally in the notebook where it is defined:

```python
filtered = gaussian_convolution(data, sigma=0.02)
```

You can also load the file from any Python session. This path does not require a
running manager:

```python
from erlab.extensions import load_script

gaussian_tools = load_script("/path/to/gaussian_tools.py")
filtered = gaussian_tools.gaussian_convolution(data, sigma=0.02)
```

{func}`erlab.extensions.routine` adds metadata to the function without wrapping it.
The manager validates the function when you enable the script.

### Routine signature rules

A routine must meet these rules:

- The first parameter is required and annotated as {class}`xarray.DataArray`.
- The return value is annotated as {class}`xarray.DataArray`.
- The remaining parameters use `bool`, `int`, `float`, `str`,
  {class}`pathlib.Path`, {data}`typing.Literal`, an {class}`enum.Enum`, or an
  optional form of one of these types.
- The function does not use `*args`, `**kwargs`, or positional-only user parameters.
- The function returns a new result. The manager presents input NumPy arrays as
  read-only data.

The GUI makes an editor for each supported user parameter. A `Literal` or `Enum`
becomes a choice control. An optional parameter can also use `None`.

The routine ID defaults to the Python function name. If you expect to rename the
function, set a stable ID before you publish the script:

```python
@routine(id="normalize", name="Normalize", category="My Lab")
def normalize_data(data: xr.DataArray) -> xr.DataArray:
    return data / data.max()
```

Do not change an existing ID without a migration plan. Saved workspaces use it to
identify the routine.

Extension routines run outside the Qt GUI thread. Do not access Qt widgets or manager
objects from an extension function.

## Add and run a script in the GUI

To register a script:

1. Select {menuselection}`Extensions --> Add Script…`.
2. Select the `.py` file.
3. Review its source and continue only if you trust it.
4. The manager imports the script, validates all decorated functions, and enables the
   extension.

The script file name is its visible extension name. For example,
`gaussian_tools.py` appears as `gaussian_tools.py`.

To run a routine:

1. Select exactly one ImageTool row.
2. Select the routine from {menuselection}`Extensions`, the ImageTool row context
   menu, or {menuselection}`Extensions --> Run Routine…`.
3. Set its parameters and run it.

The result opens as a new ImageTool row. ERLabPy records the extension call and its
parameters in provenance. The routine actions are unavailable when zero or multiple
ImageTool rows are selected.

Use {menuselection}`Extensions --> Run Routine…` to add or remove a routine from
Favorites. Script registrations and script enablement apply to all active manager
windows. Changes made in one manager appear in the other managers automatically.

## Manage registered scripts

Open {menuselection}`Extensions --> Manage Extensions` to search registered scripts,
enable or disable them, inspect their routines and loaders, and see where each source
is stored.

For a script, the details pane shows two paths:

- **Registered source file** is the file that you edit and the manager executes.
- **Stored recovery source** is a private copy of the approved source. The manager
  uses it for recovery and workspace preservation. It is not an implicit execution
  location.

If you edit the registered file, the manager stops using it until you select
{guilabel}`Review Update…` and approve the new source. Reloading unchanged contents does
not create a new source record.

If an enabled registered file is missing, the manager prompts you when it starts. You
can select {guilabel}`Locate Script…` to register an identical file at a new location,
or select {guilabel}`Restore Stored Copy…` to save the recovery source as a new file.
The manager never silently executes the stored copy.

{guilabel}`Remove Extension…` removes a script registration and its unshared recovery
source. It does not delete the registered source file. Removal is unavailable while
another manager is running, while a routine or loader call is active or waiting, or
while the current workspace depends on it.

:::{note}

The manager does not add the script directory to `sys.path`. A script can import ERLabPy
and other installed dependencies, but it should not depend on sibling files through an
implicit local import.

In a standalone packaged application, scripts can use the dependencies bundled with
the application. In a manager started from a Python environment, scripts can use the
dependencies installed in that environment. The manager does not install missing
dependencies.

:::

## Share a script

Share the Python file as the authoritative extension source. A colleague can register
the same file with {menuselection}`Extensions --> Add Script…`. The local catalog stores
the registered path, approval, script enablement, embedding policy, and Favorites. It
is not a substitute for source control.

Workspaces provide a separate reproducibility path. By default, a workspace embeds the
exact source for each script that its data or provenance uses. Embedded source is not
installed and is never imported when the workspace opens.

When a workspace contains an embedded script:

- If the same source is already registered, the manager resolves it without a prompt.
- If it has no local registration, saved data still opens. Use
  {menuselection}`Extensions --> Workspace Requirements`, select the script, and select
  {guilabel}`Save and Register Script…` before you replay its operations.
- If the same extension name is registered with different source, the manager saves
  the embedded script under a different suggested name, registers it separately, and
  updates the open workspace to use that registration. It does not replace the local
  script.
- If the required source cannot be made available, saved data and generic provenance
  remain visible. Replay and refresh actions that need the source remain unavailable.
  Save the recovered document with **Save Workspace As** so that the original file is
  not overwritten.

Code generation is unavailable for a script that exists only as an embedded workspace
object. After you save and register the script, copied code uses the public
{func}`erlab.extensions.load_script` API and the registered file path. It uses the
current local script rather than an internal manager function or an embedded object.

### Choose what a workspace embeds

The script details in {guilabel}`Manage Extensions` provide three policies:

- **Embed when referenced** is the default. The workspace embeds the script when a
  saved operation or file load uses it.
- **Always embed** includes the script even when the current workspace does not use it.
- **Never embed** stores the dependency but not the source. Use this only when every
  user can obtain the script separately.

Environment package extensions are recorded by reference and are not embedded. A
workspace records the distribution name and version, entry point, capability, and
source identity that it used. If the package is missing, saved data still opens but
dependent replay and refresh actions remain unavailable. The Workspace Requirements
dialog can copy a standard pinned install requirement. ERLab does not install it.

## Write a simple file loader

Use {func}`erlab.extensions.loader` for a path-based loader that does not need the full
{class}`erlab.io.dataloader.LoaderBase` interface:

```python
from pathlib import Path

import numpy as np
import xarray as xr

from erlab.extensions import loader


@loader(
    name="Lab text matrix",
    category="My Lab",
    summary="Load a numeric text matrix.",
    extensions=".txt",
)
def load_lab_text(path: Path, delimiter: str = ",") -> xr.DataArray:
    values = np.atleast_2d(np.loadtxt(path, delimiter=delimiter))
    return xr.DataArray(values, dims=("row", "column"))
```

A decorated loader must accept a {class}`pathlib.Path` as its first parameter. It must
return an {class}`xarray.DataArray`, {class}`xarray.Dataset`, or
{class}`xarray.DataTree`. Its remaining parameters follow the same rules as routine
parameters.

After you register and enable the script, the loader is available from manager file
open dialogs, drag and drop, recent loaders, the Data Explorer,
{func}`erlab.interactive.imagetool.manager.load_in_manager`, and file provenance.

The function is also a normal Python function. Alternatively, load it from its script:

```python
from pathlib import Path

from erlab.extensions import load_script

lab_loaders = load_script("/path/to/lab_loaders.py")
data = lab_loaders.load_lab_text(Path("scan.txt"), delimiter="\t")
```

## Publish an extension in a Python package

A package can expose decorated routines and loaders through the `erlab.extensions`
entry-point group. For example, add this table to `pyproject.toml`:

```toml
[project.entry-points."erlab.extensions"]
my_lab = "my_lab.extensions"
```

The target can be a public module that contains decorated functions or one decorated
function. Keep functions in a public import path so that copied code can use a direct
Python import.

For a full ERLab data loader, expose a
{class}`LoaderBase <erlab.io.dataloader.LoaderBase>` class through the
`erlab.io.loaders` group:

```toml
[project.entry-points."erlab.io.loaders"]
my_lab = "my_lab.loader:MyLoader"
```

See {ref}`implementing-plugins` for the `LoaderBase` interface.

The manager discovers package entry points at startup. A manager started from a normal
Python environment discovers packages in that environment. A standalone manager
discovers packages that its application builder included in the application, together
with their distribution metadata. It does not inspect packages from another Python
environment.

After you install, remove, or edit a package in a normal Python environment while the
manager is open, open
{menuselection}`Extensions --> Manage Extensions` and select
{guilabel}`Refresh Environment Packages`. Discovered package extensions are available
automatically after validation. The manager does not provide Enable, Disable, Remove,
or approval actions for packages. A standalone application does not show the refresh
action because its bundled packages are fixed when the application is built.

The Python environment is the package catalog and package manager. ERLab does not copy
package records into its persistent script catalog. Package import results and
capability descriptors exist only for the current manager process. Install, update, or
uninstall a package with the normal tools for that Python environment.

Package installation, dependency installation, and wheel management are outside the
extension manager. Use your normal packaging and environment tools for these tasks.

## Diagnose an extension

The {guilabel}`Manage Extensions` status separates script enablement from source
health. A script can be disabled but otherwise ready, waiting for approval, missing,
changed, or failing to import. A package is available when its discovered entry point
loads successfully. Select {guilabel}`Show Error Details` when an import fails.

An extension import or call is also written to the manager log with the extension,
capability, input shape, output shape, duration, final status, and a traceback on
failure. The manager shows one concise error dialog for a failed call.

Common failures have these causes:

- **Unsupported signature**: add the required annotations and use supported parameter
  types.
- **Import failed**: install the dependency in the active Python environment, or use a
  dependency bundled with the standalone manager.
- **Source unavailable**: locate or restore the registered script.
- **Source changed**: review and register the current file contents.
- **Workspace requirement unavailable**: open {guilabel}`Workspace Requirements` and
  save and register an available embedded script.

See {mod}`erlab.extensions` for the complete public API and error types.
