(imagetool-manager-extensions)=

# ImageTool Manager extensions

You can incorporate custom analysis routines and file loaders into ImageTool Manager as
*extensions*.

An ImageTool Manager extension is one Python script that contains one or more decorated
functions.

The decorators are {func}`erlab.extensions.routine` and {func}`erlab.extensions.loader`,
depending on whether the function is an analysis routine or a file loader. A decorated
function is still a normal Python function, so you can use it in a notebook.

## Implementing an extension

### Write a routine

Create a new Python file named `gaussian_tools.py` that contains the following code:

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

The first parameter is the selected ImageTool data. The other parameters become controls
in the routine dialog. The return value opens in a new ImageTool.

#### Test the routine

Call the function directly when you work in the same notebook or module:

```python
filtered = gaussian_convolution(data, sigma=0.02)
```

Use {func}`erlab.extensions.load_script` to test the saved file from another Python
session:

```python
from erlab.extensions import load_script

gaussian_tools = load_script("/path/to/gaussian_tools.py")
filtered = gaussian_tools.gaussian_convolution(data, sigma=0.02)
```

#### Routine signature rules

A routine must follow these rules:

- The first parameter must be annotated as {class}`xarray.DataArray`.
- The return value must be annotated as {class}`xarray.DataArray`.
- Every parameter must have a supported annotation.
- Do not use positional-only parameters, `*args`, or `**kwargs`.
- Asynchronous functions are not supported.
- Do not mutate the input data.

You can use these parameter types:

- `bool`
- `int`
- `float`
- `str`
- {class}`pathlib.Path`
- {data}`typing.Literal`
- {class}`enum.Enum`
- An optional form of the above types, e.g., `float | None`.

A `Literal` or `Enum` parameter becomes a choice control.

The routine ID is the Python function name by default. Explicitly supply `id` if you
want to rename the function later without changing its identity:

```python
@routine(id="normalize", name="Normalize", category="My Lab")
def normalize_data(data: xr.DataArray) -> xr.DataArray:
    return data / data.max()
```

Since saved operations use the routine ID, the ID must be kept stable to maintain
reproducibility.

### Write a file loader

Use a loader for a simple file format that returns one xarray object:

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

A loader must follow these rules:

- The first parameter must have the {class}`pathlib.Path` annotation.
- The return value must have an {class}`xarray.DataArray`,
  {class}`xarray.Dataset`, or {class}`xarray.DataTree` annotation.
- Each other parameter must follow the routine parameter rules.
- A leading period in each `extensions` value is optional.
- Do not use `loader_extensions` or `without_values` as a parameter name.

You can test the loader as a normal function. You can also load it from its script:

```python
from pathlib import Path

from erlab.extensions import load_script

lab_loaders = load_script("/path/to/lab_loaders.py")
data = lab_loaders.load_lab_text(Path("scan.txt"), delimiter="\t")
```

## Register and use the script

1. Start ImageTool Manager.
2. Select {menuselection}`Extensions --> Add Script…`.
3. Select the `.py` file.
4. Review the source and approve.

Each registered script must have a unique file name. ImageTool Manager compares file
names without case differences. For example, you cannot register both
`gaussian_tools.py` and `GAUSSIAN_TOOLS.py`.

To run a routine, select one ImageTool and then select the routine from the
{menuselection}`Extensions` menu. To use a loader, open a file with the file filter that
the loader supplies.

After you edit a registered script, open {menuselection}`Extensions --> Manage
Extensions` and review the update. The manager does not run changed code before you
approve it.

If you move a registered script, ImageTool Manager asks you to locate the file. The new
file must have the same file name and the same contents.

## Use a workspace with an extension

By default, ImageTool Manager stores the exact contents of each script that the
workspace uses. This copy is for recovery and reproducibility. The manager never
runs this embedded copy.

You can change the workspace embedding setting in {menuselection}`Extensions --> Manage
Extensions`. Use **Always embed** to include an unused script. Use **Never embed** when
you manage recovery by another method.

If a script used in the manager is not available, save the embedded copy to a local
`.py` file and register it before you replay the operation.

## Dependencies

An extension script can import ERLabPy and other available Python packages. For
example:

```python
import xarray as xr

from erlab.extensions import routine
from some_package import do_something


@routine(name="Remove background", category="My Lab")
def remove_background(data: xr.DataArray) -> xr.DataArray:
    return do_something(data)
```

`some_package` must be installed in the Python environment that starts ImageTool
Manager. The standalone version cannot access these, so you must use a Python
environment to run the manager if your extension depends on external packages or build a
standalone version that includes the packages.

Do not depend on an implicit import from a file next to the extension script. The
manager does *not* add the script directory to the Python import path.

## Share the extension

- Share the `.py` file as the extension source. Using a version control system like git
  is recommended if you want to track changes and share the extension with other users.
  Each user should add their local copy with {menuselection}`Extensions --> Add Script…`.

- If an import fails, confirm that all imported packages are available to the manager.

- If validation fails, confirm that each function follows the signature rules above.

See {mod}`erlab.extensions` for the complete public API and error types.
