(imagetool-manager-extensions)=

# Write an ImageTool Manager extension

An ImageTool Manager extension is one Python script. The script contains one or more
decorated functions. You do not need to use Qt or ImageTool Manager classes.

Use {func}`erlab.extensions.routine` to add an analysis routine. Use
{func}`erlab.extensions.loader` to add a file loader. A decorated function is still a
normal Python function, so you can test it in a notebook.

:::{warning}

An extension is Python code. Add a script only if you trust its source.

:::

## Write a routine

Save this example as `gaussian_tools.py`:

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

The first parameter is the selected ImageTool data. The other parameters become
controls in the routine dialog. The return value opens in a new ImageTool.

The `sigma` value in this example uses coordinate units, not pixel units.

### Test the routine

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

This code does not require a running ImageTool Manager.

### Follow the routine signature rules

A routine must follow these rules:

- The first parameter must have the {class}`xarray.DataArray` annotation.
- The return value must have the {class}`xarray.DataArray` annotation.
- Each other parameter must have a supported annotation.
- Do not use positional-only parameters, `*args`, or `**kwargs`.
- Use a synchronous function. Do not use `async def` or `yield`.
- Do not use `erlab` as the decorated function name. The loaded script uses
  `erlab` for script information.
- Return a result. Do not modify the input data.

You can use these parameter types:

- `bool`
- `int`
- `float`
- `str`
- {class}`pathlib.Path`
- {data}`typing.Literal`
- {class}`enum.Enum`
- An optional form of one of these types

A `Literal` or `Enum` parameter becomes a choice control. An optional parameter can
also use `None`.

The routine ID is the Python function name by default. Set `id` if you want to rename
the function later without changing its identity:

```python
@routine(id="normalize", name="Normalize", category="My Lab")
def normalize_data(data: xr.DataArray) -> xr.DataArray:
    return data / data.max()
```

Keep a published ID stable. Saved operations use the ID to find the routine.

## Write a file loader

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
4. Review the source.
5. Approve the script.

The script file name is the extension name. For example, `gaussian_tools.py` appears
as `gaussian_tools.py`.

To run a routine, select one ImageTool and then select the routine from the
{menuselection}`Extensions` menu. To use a loader, open a file with the file filter
that the loader supplies.

After you edit a registered script, open {menuselection}`Extensions --> Manage
Extensions` and review the update. The manager does not run changed code before you
approve it.

## Use dependencies

An extension script can import ERLabPy and other available Python packages. For
example, the script can provide a short interface to a routine in a lab package:

```python
import xarray as xr

from erlab.extensions import routine
from my_lab.analysis import remove_background


@routine(name="Remove background", category="My Lab")
def remove_lab_background(data: xr.DataArray) -> xr.DataArray:
    return remove_background(data)
```

ImageTool Manager does not install dependencies. A manager started from a Python
environment can use the packages in that environment. A standalone manager can use
only the packages that are part of the application build.

ImageTool Manager does not discover extension packages. Use a small registered script
to expose functions from an installed package.

Do not depend on an implicit import from a file next to the extension script. The
manager does not add the script directory to the Python import path.

## Share the extension

Share the `.py` file as the extension source. Use version control if the extension is
maintained by a group. Each user adds their local copy with
{menuselection}`Extensions --> Add Script…`.

If an import fails, confirm that all imported packages are available to the manager.
If validation fails, confirm that each function follows the signature rules above.

See {mod}`erlab.extensions` for the complete public API and error types.
