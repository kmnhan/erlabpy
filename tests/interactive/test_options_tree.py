import math
import types

from erlab.interactive._options.schema import AppOptions
from erlab.interactive._options.tree import (
    _build_leaf_param,
    _limits_from_schema,
    make_parameter,
    parameter_to_options,
)


def test_limits_from_schema_defaults_to_infinities() -> None:
    assert _limits_from_schema({}) is None
    assert _limits_from_schema({"minimum": 1.5}) == (1.5, math.inf)
    assert _limits_from_schema({"maximum": 2.0}) == (-math.inf, 2.0)


def test_build_leaf_param_handles_enum_and_unknown_type() -> None:
    field_info = types.SimpleNamespace(json_schema_extra=None)

    enum_param = _build_leaf_param(
        "choice",
        {"enum": ["a", "b"]},
        field_info,
        value="a",
        default_value="a",
    )
    assert enum_param["type"] == "list"
    assert enum_param["limits"] == ["a", "b"]

    unknown_param = _build_leaf_param(
        "custom",
        {"type": "mystery"},
        field_info,
        value="val",
        default_value="val",
    )
    assert unknown_param["type"] == "str"
    assert unknown_param["value"] == "val"


def test_make_parameter_defaults_and_missing_child_continue() -> None:
    param = make_parameter()
    # Drop a child to exercise the continue path in parameter_to_options
    param.removeChild(param.child("colors"))
    opts = parameter_to_options(param)
    assert isinstance(opts, AppOptions)
    assert opts.colors == AppOptions().colors
