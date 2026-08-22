(imagetool-manager-extensions)=

# Manager extensions

Manager extensions add lab-specific analysis routines and file loaders to ImageTool
Manager. An extension is one Python script that contains one or more decorated
functions. A decorated function remains a normal Python function that you can call
from a notebook or script.

Use the {doc}`Python task guide <../../how-to/python/manager-extensions>` to write and
test an extension. Use the {doc}`GUI task guide <../../how-to/gui/extensions>` to
register, run, update, and recover extension scripts.

## Capability types

| Capability | Decorator | Input | Result | Manager interface |
| --- | --- | --- | --- | --- |
| Analysis routine | {func}`~erlab.extensions.routine` | One {class}`~xarray.DataArray` | One {class}`~xarray.DataArray` | An entry in the {menuselection}`Extensions` menu |
| File loader | {func}`~erlab.extensions.loader` | One {class}`~pathlib.Path` | A {class}`~xarray.DataArray`, {class}`~xarray.Dataset`, or {class}`~xarray.DataTree` | A file filter in Manager file dialogs and Data Explorer |

The parameters after the input become controls in a Manager dialog. The decorator
metadata sets the visible name, category, summary, and stable capability ID.

## Function signatures

All extension functions have these requirements:

- The function must be synchronous.
- The first parameter must be a required positional parameter. It can be positional-only
  or positional-or-keyword.
- Parameters after the first parameter cannot be positional-only. They can be
  positional-or-keyword or keyword-only.
- The function cannot use `*args` or `**kwargs`.
- All parameters and the return value must have resolvable type annotations.
- The function name `erlab` is reserved for loaded-script metadata.

An analysis routine has these additional requirements:

- The first parameter annotation must be {class}`xarray.DataArray`.
- The return annotation must be {class}`xarray.DataArray`.
- The function must not mutate the input array. Manager supplies an isolated wrapper
  and makes NumPy-backed values and coordinates read-only.

A file loader has these additional requirements:

- The first parameter annotation must be {class}`pathlib.Path`.
- The return annotation must be {class}`xarray.DataArray`, {class}`xarray.Dataset`,
  {class}`xarray.DataTree`, or a union of these types.
- The parameter names `loader_extensions` and `without_values` are reserved.
- A filename extension supplied to {func}`~erlab.extensions.loader` can include or omit
  its leading period. ERLabPy stores the normalized value with the period.

## Parameter annotations

| Python annotation | Manager control |
| --- | --- |
| `bool` | Check box |
| `int` | Integer input |
| `float` | Numeric input |
| `str` | Text input |
| {class}`pathlib.Path` | Path input |
| {data}`typing.Literal` | Choice input |
| {class}`enum.Enum` | Choice input |
| An optional form of a supported type, such as `float | None` | The corresponding input with a `None` state |

`Literal` values and `Enum` member values must be Boolean, integer, finite
floating-point, or string values. A default value must match its annotation. Only an
optional parameter can have a default of `None`.

## Script and capability identity

| Value | Purpose | Stability requirement |
| --- | --- | --- |
| Script name | Identifies a registered `.py` file | Manager compares file names case-insensitively, and each name must be unique. |
| Capability ID | Identifies one routine or loader in a script | The function name is the default. Set `id` before you rename a function that a saved operation uses. |
| Source hash | Identifies the exact script contents | Manager requires approval when the contents change. Recorded operations use the approved hash. |

For example, this explicit ID remains `normalize` if the Python function name changes:

```python
import xarray as xr

from erlab.extensions import routine


@routine(id="normalize", name="Normalize", category="My Lab")
def normalize_data(data: xr.DataArray) -> xr.DataArray:
    return data / data.max()
```

## Manager behavior

| State or event | Manager behavior |
| --- | --- |
| New script | Shows the source for approval before registration and validation. |
| Changed script | Does not run the changed source until you review and approve it. |
| Missing script | Requests a file with the same name and exact approved contents. |
| Routine result | Opens the returned array in a new ImageTool and records the operation. |
| Loader result | Opens the returned xarray object through the normal file-loading workflow. |
| Workspace source | Stores an exact source copy according to the workspace embedding policy. It never runs the embedded copy directly. |

The {guilabel}`Workspace embedding` setting has these values:

| Value | Source stored in the workspace |
| --- | --- |
| {guilabel}`Embed when referenced` | Scripts required by recorded workspace operations. This is the default. |
| {guilabel}`Always embed` | The script, even when no saved operation uses it. |
| {guilabel}`Never embed` | No copy of the script. Recovery depends on the external file. |

## Python environment

An extension runs in the Python environment that starts ImageTool Manager. The script
can import ERLabPy and packages in that environment. The standalone Manager can import
only its bundled packages.

Manager does not add the extension script directory to the Python import path. Put
shared code in an installed package instead of depending on an implicit import from a
file beside the extension script.

## Public API

The {mod}`erlab.extensions` API Reference describes the decorators, script loader,
execution functions, descriptors, and error types.
