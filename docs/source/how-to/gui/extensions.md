(how-to-gui-manager-extensions)=

# Manager extensions

Use these procedures to create, register, and maintain lab-specific analysis routines
and file loaders for ImageTool Manager. Review an extension as executable Python code
before you approve it.

To write an extension, you must be able to create a Python module and have
representative test data.

For extension types, signature rules, and workspace states, see
{ref}`imagetool-manager-extensions`.

## Writing an analysis routine

Save the following code as `gaussian_tools.py`:

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

The first parameter receives the selected ImageTool data. Manager creates controls for
`sigma` and `mode`. It opens the returned array in a new ImageTool.

Call the decorated function directly with an existing {class}`~xarray.DataArray` named
`data`:

```python
from gaussian_tools import gaussian_convolution

filtered = gaussian_convolution(data, sigma=0.02)
```

Then validate the saved source independently of the normal module import:

```python
from erlab.extensions import load_script

gaussian_tools = load_script("gaussian_tools.py")
filtered = gaussian_tools.gaussian_convolution(data, sigma=0.02)
```

{func}`~erlab.extensions.load_script` imports the saved source and validates all its
decorated functions. If validation fails, use the reported function and parameter name
to correct the signature.

## Preserving a capability ID

Before you rename a function that saved workspaces use, set `id` to its existing
function name:

```python
@routine(id="normalize", name="Normalize", category="My Lab")
def normalize_data(data: xr.DataArray) -> xr.DataArray:
    return data / data.max()
```

This example keeps the capability ID `normalize` after the function is renamed to
`normalize_data`.

## Writing a file loader

Use a loader when one function can read the file and return one xarray object. Save the
following code as `lab_loaders.py`:

```python
from pathlib import Path

import numpy as np
import xarray as xr

from erlab.extensions import loader


@loader(
    name="Lab text matrix",
    category="My Lab",
    summary="Load a numeric text matrix.",
    extensions=(".txt",),
)
def load_lab_text(path: Path, delimiter: str = ",") -> xr.DataArray:
    values = np.atleast_2d(np.loadtxt(path, delimiter=delimiter))
    return xr.DataArray(values, dims=("row", "column"))
```

Test the saved script with a representative file:

```python
from pathlib import Path

from erlab.extensions import load_script

lab_loaders = load_script("lab_loaders.py")
loaded = lab_loaders.load_lab_text(Path("scan.txt"), delimiter="\t")
```

Use {doc}`../../contributing/loaders` when the format needs scan identification,
metadata normalization, multiple-file assembly, or reusable loader configuration.

## Adding dependencies

Import a dependency in the extension script as you would in a normal module:

```python
import xarray as xr

from erlab.extensions import routine
from some_package import do_something


@routine(name="Remove background", category="My Lab")
def remove_background(data: xr.DataArray) -> xr.DataArray:
    return do_something(data)
```

Install `some_package` in the environment that starts Manager. If you use the
standalone Manager, use only bundled packages or {ref}`build a standalone application
<build-from-source>` that includes the dependency.

Put shared modules in an installed package. Manager does not add the extension script
directory to the Python import path, so an implicit import from a neighboring file does
not work.

(how-to-gui-register-extension)=

## Registering a script

1. Start ImageTool Manager.
2. Select {menuselection}`Extensions --> Add Script…`.
3. Select the `.py` file.
4. Review the complete source.
5. Select {guilabel}`OK` to approve and register it.

Each registered script must have a unique file name. Manager compares file names
without case differences. For example, you cannot register both `gaussian_tools.py`
and `GAUSSIAN_TOOLS.py`.

## Running an analysis routine

1. Select one ImageTool in Manager.
2. Select the routine from the {menuselection}`Extensions` menu or the selected row's
   {menuselection}`Extensions` submenu.
3. Enter the routine parameters.
4. Select {guilabel}`OK`.

Manager opens the result in a new ImageTool and records the extension operation in its
provenance.

## Loading a file

1. Open a Manager file dialog or Data Explorer.
2. Select the file filter supplied by the extension loader.
3. Select the file and enter any loader parameters.
4. Open the file.

Use {ref}`how-to-gui-manager-open` for drag-and-drop, batch loading, and Data Explorer
workflows.

## Approving a script update

After you edit a registered script, Manager stops running that changed source until you
approve it.

1. Select {menuselection}`Extensions --> Manage Extensions`.
2. Select the script with the {guilabel}`Approval required` state.
3. Select {guilabel}`Review Update…`.
4. Review the complete source and select {guilabel}`OK`.

If validation fails, select {guilabel}`Show Error Details`. Correct the source file,
then review the update again.

## Locating a moved script

If Manager reports that a registered script is missing:

1. Select the script in the {guilabel}`Extension Scripts Not Found` dialog.
2. Select {guilabel}`Locate Script…`.
3. Select the script at its new location.

The selected file must have the same file name and the same contents as the approved
script. To use changed contents, restore the approved file first, then follow the
script update procedure.

## Selecting workspace embedding

1. Select {menuselection}`Extensions --> Manage Extensions`.
2. Select the script.
3. Set {guilabel}`Workspace embedding` to one of these values:

   - {guilabel}`Embed when referenced` stores the script when a saved operation uses
     it.
   - {guilabel}`Always embed` stores the script even when no saved operation uses it.
   - {guilabel}`Never embed` omits the source from the workspace.

Use {guilabel}`Never embed` only when another recovery method preserves the exact
approved source.

(how-to-gui-recover-extension)=

## Recovering a script from a workspace

If a workspace contains an embedded copy of an unavailable extension:

1. Select {menuselection}`Extensions --> Workspace Requirements`.
2. Select the unavailable extension.
3. Select {guilabel}`Save and Register Script…`.
4. Review the embedded source.
5. Save it as a local `.py` file.

Manager registers the saved file and updates the workspace requirement. It never runs
the embedded source directly.

See {doc}`workspaces and provenance
<../../explanation/workspaces-and-provenance>` for the source-trust and replay model.

## Sharing an extension

Share the `.py` source file. Use a version control system when several users must track
changes to the same script. Each user registers a local copy in Manager.

## Troubleshooting a script

If the script does not load:

- Run {func}`~erlab.extensions.load_script` to display import and signature errors.
- Confirm that each imported package is in the Manager environment.
- Compare the decorated functions with the {ref}`signature requirements
  <imagetool-manager-extensions>`.
