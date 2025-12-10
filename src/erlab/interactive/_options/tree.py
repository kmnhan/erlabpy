"""Dynamic ParameterTree generation from Pydantic models.

`make_parameter` introspects the JSON Schema of `AppOptions` to build a
pyqtgraph Parameter tree. Field metadata (title, description, limits, enum,
custom UI hints) comes from Pydantic `Field` definitions and `json_schema_extra`.
"""

from __future__ import annotations

import typing

import pyqtgraph.parametertree
from pydantic import BaseModel

from erlab.interactive._options.schema import AppOptions


def _field_ui_type(field_info) -> str | None:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    return extra.get("ui_type") if isinstance(extra, dict) else None


def _limits_from_schema(s: dict[str, typing.Any]) -> tuple[float, float] | None:
    lo = s.get("minimum")
    hi = s.get("maximum")
    if lo is None and hi is None:
        return None
    if lo is None:
        lo = float("-inf")
    if hi is None:
        hi = float("inf")
    return (lo, hi)


def _build_leaf_param(
    name: str,
    schema: dict[str, typing.Any],
    field_info,
    value: typing.Any,
    default_value: typing.Any,
) -> dict[str, typing.Any]:
    title = schema.get("title", name)
    desc = schema.get("description")
    ui_type: str | None = _field_ui_type(field_info)
    param_type: str
    limits = None
    opts: dict[str, typing.Any] = {}

    if "enum" in schema:
        param_type = "list"
        opts["limits"] = schema["enum"]
    else:
        stype = schema.get("type")
        if ui_type:
            param_type = ui_type
        else:
            match stype:
                case "boolean":
                    param_type = "bool"
                case "number" | "integer":
                    param_type = "float" if stype == "number" else "int"
                    limits = _limits_from_schema(schema)
                    if limits is not None:
                        opts["limits"] = limits

                case "array":
                    # represent as CSV string unless ui_type specified
                    param_type = "str"
                    if isinstance(value, list):
                        value = ", ".join(value)
                        default_value = ", ".join(default_value)
                case _:
                    param_type = "str"

    extras = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extras, dict):
        for k, v in extras.items():
            if k.startswith("ui_") and k != "ui_type":
                opts[k.removeprefix("ui_")] = v

    param: dict[str, typing.Any] = {
        "name": name,
        "title": title,
        "type": param_type,
        "value": value,
        "default": default_value,
    }
    if desc:
        param["tip"] = desc
    param.update(opts)
    return param


def _build_group(
    model_instance: BaseModel, model_cls: type[BaseModel]
) -> list[dict[str, typing.Any]]:
    schema: dict[str, typing.Any] = model_cls.model_json_schema()
    defaults: BaseModel = model_cls()
    props: dict[str, typing.Any] = schema.get("properties", {})
    children: list[dict[str, typing.Any]] = []
    for fname, field in model_cls.model_fields.items():
        fschema = props.get(fname, {})
        val = getattr(model_instance, fname)
        def_val = getattr(defaults, fname)
        if isinstance(val, BaseModel):
            group_children = _build_group(val, type(val))
            children.append(
                {
                    "name": fname,
                    "title": fschema.get("title", fname.capitalize()),
                    "type": "group",
                    "children": group_children,
                    "tip": fschema.get("description"),
                }
            )
        else:
            children.append(_build_leaf_param(fname, fschema, field, val, def_val))
    return children


def make_parameter(
    options: AppOptions | None = None,
) -> pyqtgraph.parametertree.Parameter:
    if options is None:
        options = AppOptions()
    return pyqtgraph.parametertree.Parameter.create(
        name="Settings",
        children=_build_group(options, AppOptions),
    )


def parameter_to_options(param: pyqtgraph.parametertree.Parameter) -> AppOptions:
    """Rebuild an `AppOptions` instance from a Parameter tree."""
    T = typing.TypeVar("T", bound=BaseModel)

    def _extract(model_cls: type[T], group_param) -> T:
        data: dict[str, typing.Any] = {}
        default_instance = model_cls()
        for fname in model_cls.model_fields:
            try:
                child = group_param.child(fname)
            except KeyError:
                continue
            val = child.value() if child.hasChildren() is False else None
            # Recurse for nested models
            nested_candidate = getattr(default_instance, fname)
            if isinstance(nested_candidate, BaseModel):
                nested_group = group_param.child(fname)
                data[fname] = _extract(type(nested_candidate), nested_group)
                continue
            data[fname] = val
        return model_cls.model_validate(data)

    return _extract(AppOptions, param)
