(how-to-python-manager-extensions)=

# Writing Manager extensions

Use these procedures to add a lab-specific analysis routine or file loader to
ImageTool Manager. They assume that you can create a Python module and that you have
your own test data.

See the {ref}`Manager extension reference <imagetool-manager-extensions>` for all
supported signatures and parameter annotations.

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

Use a full {doc}`loader plugin <../../contributing/loaders>` when the format needs scan
identification, metadata normalization, multiple-file assembly, or reusable loader
configuration.

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
standalone Manager, use only bundled packages or {doc}`build a standalone application
<../../contributing/build-manager>` that includes the dependency.

Put shared modules in an installed package. Manager does not add the extension script
directory to the Python import path, so an implicit import from a neighboring file does
not work.

## Sharing an extension

Share the `.py` source file. Use a version control system when several users must track
changes to the same script. Each user registers a local copy in Manager.

If the script does not load:

- Run {func}`~erlab.extensions.load_script` to display import and signature errors.
- Confirm that each imported package is in the Manager environment.
- Compare the decorated functions with the {ref}`signature requirements
  <imagetool-manager-extensions>`.

Continue with {ref}`registering the script <how-to-gui-register-extension>`.
