import importlib
import pathlib

import xarray as xr

import erlab.accessors
from erlab.accessors.utils import ERLabDataArrayAccessor, ERLabDatasetAccessor


def _find_accessor_name(parent_class, accessor_class) -> str | None:
    parent_instance = parent_class()

    for attr_name in dir(parent_instance):
        try:
            attr = getattr(parent_instance, attr_name)
        except AttributeError:
            continue
        else:
            if isinstance(attr, accessor_class):
                return attr_name
    return None


content_da_callable: str = ""
content_da_methods: str = ""
content_da_attributes: str = ""
content_ds_callable: str = ""
content_ds_methods: str = ""
content_ds_attributes: str = ""

kspace_methods: str = ""
kspace_attributes: str = ""


def _make_table_row(name, kind):
    global content_da_callable, content_ds_callable
    global content_da_methods, content_ds_methods
    global content_da_attributes, content_ds_attributes
    global kspace_methods, kspace_attributes

    row = f"\n   {name}"

    if name.startswith("DataArray"):
        if name.removeprefix("DataArray.").startswith("kspace"):
            match kind:
                case "method":
                    kspace_methods += row
                case "attribute":
                    kspace_attributes += row
        else:
            match kind:
                case "callable":
                    content_da_callable += row
                case "method":
                    content_da_methods += row
                case "attribute":
                    content_da_attributes += row
    elif name.startswith("Dataset"):
        match kind:
            case "callable":
                content_ds_callable += row
            case "method":
                content_ds_methods += row
            case "attribute":
                content_ds_attributes += row


def _make_accessor_summary(accessor, accessor_cls_name):
    if issubclass(accessor, ERLabDataArrayAccessor):
        prefix = f"DataArray.{_find_accessor_name(xr.DataArray, accessor)}"

    elif issubclass(accessor, ERLabDatasetAccessor):
        prefix = f"Dataset.{_find_accessor_name(xr.Dataset, accessor)}"
    else:
        return

    members = dir(accessor)

    for m in members:
        if m.startswith("_") and m != "__getitem__" and m != "__call__":
            continue

        if m == "__call__":
            _make_table_row(prefix, "callable")
        else:
            member_obj = getattr(accessor, m)
            _make_table_row(
                f"{prefix}.{m}", "method" if callable(member_obj) else "attribute"
            )


for path in pathlib.Path(erlab.accessors.__file__).resolve().parent.iterdir():
    if (
        path.is_file()
        and path.suffix == ".py"
        and not path.name.startswith((".", "__"))
    ):
        module_name = f"{erlab.accessors.__name__}.{path.stem}"
        module = importlib.import_module(module_name)
        if hasattr(module, "__all__"):
            for module_attr_name in module.__all__:
                accessor_cls = getattr(module, module_attr_name)
                accessor_cls_name = f"{module_name}.{module_attr_name}"

                _make_accessor_summary(accessor_cls, accessor_cls_name)


def _make_section(table_content: str, header: str) -> str:
    if not table_content:
        return ""

    return f"\n{header}\n{len(header) * '~'}\n\n.. list-table::\n{table_content}"


def _make_accessor_autosummary(
    content: str, kind: str, header=None, header_level=3
) -> str:
    if not content:
        return ""

    header = "" if header is None else f"{'#' * header_level} {header}"

    return f"""{header}

```{{eval-rst}}
.. autosummary::
   :toctree: accessors
   :template: autosummary/accessor_{kind}.rst
   {content}
```
"""


content = f"""# Accessors ({{mod}}`erlab.accessors`)

```{{eval-rst}}
.. automodule:: erlab.accessors

.. currentmodule:: xarray
```

## Dataset accessors

### Methods
{_make_accessor_autosummary(content_ds_callable, "callable")}
{_make_accessor_autosummary(content_ds_methods, "method")}

{_make_accessor_autosummary(content_ds_attributes, "attribute", "Attributes")}

## DataArray accessors

### General

#### Methods
{_make_accessor_autosummary(content_da_callable, "callable")}
{_make_accessor_autosummary(content_da_methods, "method")}

{_make_accessor_autosummary(content_da_attributes, "attribute", "Attributes")}

### Momentum Space

#### Methods
{_make_accessor_autosummary(kspace_methods, "method")}

#### Attributes
{_make_accessor_autosummary(kspace_attributes, "attribute")}

"""

with open("erlab.accessors.md", "w") as f:
    f.write(content)
