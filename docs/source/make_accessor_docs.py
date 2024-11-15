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


content_da_methods: str = ""
content_da_attributes: str = ""
content_ds_methods: str = ""
content_ds_attributes: str = ""


def _make_table_row(name, obj, obj_full_name: str):
    global \
        content_da_methods, \
        content_da_attributes, \
        content_ds_methods, \
        content_ds_attributes

    directive = ":meth:" if callable(obj) else ":attr:"

    doc = obj.__doc__
    if doc:
        doc = doc.strip().split("\n")[0]

    row = f"\n   * - {directive}`{name} <{obj_full_name}>`\n     - {doc}"

    if name.startswith("DataArray"):
        if callable(obj):
            content_da_methods += row
        else:
            content_da_attributes += row
    elif name.startswith("Dataset"):
        if callable(obj):
            content_ds_methods += row
        else:
            content_ds_attributes += row


def _make_accessor_summary(accessor, accessor_cls_name):
    if issubclass(accessor, ERLabDataArrayAccessor):
        prefix = f"DataArray.{_find_accessor_name(xr.DataArray, accessor)}"

    elif issubclass(accessor, ERLabDatasetAccessor):
        prefix = f"Dataset.{_find_accessor_name(xr.Dataset, accessor)}"
    else:
        return

    members = dir(accessor)

    if "__call__" in members:
        _make_table_row(prefix, accessor.__call__, f"{accessor_cls_name}.__call__")

    for m in members:
        if m.startswith("_") and m != "__getitem__":
            continue
        member_obj = getattr(accessor, m)
        member_obj_name = f"{accessor_cls_name}.{m}"
        _make_table_row(f"{prefix}.{m}", member_obj, member_obj_name)


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


content = f"""Accessors (:mod:`erlab.accessors`)
==================================

.. automodule:: erlab.accessors

Dataset accessors
-----------------
{_make_section(content_ds_methods, 'Methods')}
{_make_section(content_ds_attributes, 'Attributes')}

DataArray accessors
-------------------
{_make_section(content_da_methods, 'Methods')}
{_make_section(content_da_attributes, 'Attributes')}

"""

with open("erlab.accessors.rst", "w") as f:
    f.write(content)
