import math
import types

from erlab.interactive._file_loaders import BUILTIN_FILE_LOADER_SPECS
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


def test_make_parameter_round_trips_figure_options() -> None:
    options = AppOptions.model_validate(
        {
            "figure": {
                "stylesheets": ["classic", "missing-style"],
                "dpi": 150.0,
                "export": {
                    "dpi": 300.0,
                    "transparent": "true",
                    "bbox_inches": "tight",
                    "pad_inches": "layout",
                },
            }
        }
    )
    param = make_parameter(options)

    stylesheet_param = param.child("figure").child("stylesheets")
    assert stylesheet_param.opts["type"] == "matplotlib_stylesheets"
    assert stylesheet_param.value() == ["classic", "missing-style"]
    dpi_param = param.child("figure").child("dpi")
    assert dpi_param.opts["type"] == "figure_dpi_override"
    assert dpi_param.value() == 150.0
    export_param = param.child("figure").child("export")
    assert export_param.child("dpi").opts["type"] == "savefig_dpi"
    assert export_param.child("dpi").value() == 300.0
    assert export_param.child("transparent").opts["limits"] == {
        "Use stylesheet": "style",
        "Enabled": "true",
        "Disabled": "false",
    }
    assert export_param.child("bbox_inches").value() == "tight"
    assert export_param.child("pad_inches").opts["type"] == "savefig_padding"
    assert export_param.child("pad_inches").value() == "layout"

    opts = parameter_to_options(param)
    assert opts.figure.stylesheets == ["classic", "missing-style"]
    assert opts.figure.dpi == 150.0
    assert opts.figure.export == options.figure.export


def test_make_parameter_round_trips_default_directory() -> None:
    options = AppOptions.model_validate({"io": {"default_directory": "~/data"}})
    param = make_parameter(options)

    directory_param = param.child("io").child("default_directory")
    assert directory_param.opts["type"] == "directory_path"
    assert directory_param.value() == "~/data"

    directory_param.setValue("~/other-data")

    assert parameter_to_options(param).io.default_directory == "~/other-data"


def test_make_parameter_round_trips_builtin_default_loader() -> None:
    loader_id = BUILTIN_FILE_LOADER_SPECS[0].id
    options = AppOptions.model_validate({"io": {"default_loader": loader_id}})
    param = make_parameter(options)

    loader_param = param.child("io").child("default_loader")
    assert loader_id in loader_param.opts["limits"].values()
    assert loader_param.value() == loader_id
    assert parameter_to_options(param).io.default_loader == loader_id


def test_make_parameter_workspace_compression_uses_display_labels() -> None:
    param = make_parameter()
    compression_param = param.child("io").child("workspace").child("compression")

    assert compression_param.opts["type"] == "list"
    assert set(compression_param.opts["limits"].values()) == {
        "none",
        "blosclz3",
        "zstd1",
    }

    compression_param.setValue("blosclz3")

    assert parameter_to_options(param).io.workspace.compression == "blosclz3"
