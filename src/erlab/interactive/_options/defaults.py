"""Module that decides the default options and the structure of the options dialog."""

import typing

import pyqtgraph.parametertree

import erlab

DEFAULT_OPTIONS: dict[str, dict[str, typing.Any]] = {
    "colors": {
        "cmap": {
            "name": "magma",
            "gamma": 0.5,
            "reverse": False,
            "exclude": [
                "prism",
                "tab10",
                "tab20",
                "tab20b",
                "tab20c",
                "flag",
                "Set1",
                "Set2",
                "Set3",
                "Pastel1",
                "Pastel2",
                "Pastel3",
                "Paired",
                "Dark2",
            ],
            "packages": [
                "cmasher",
                "cmocean",
                "colorcet",
                "cmcrameri",
                "seaborn",
            ],
        },
        "cursors": [
            "#cccccc",
            "#ffff00",
            "#ff00ff",
            "#00ffff",
            "#008000",
            "#ff0000",
            "#0000ff",
        ],
    },
    "io": {
        "default_loader": None,
    },
}  #: Default configuration for the ImageTool, hardcoded.


def make_parameter(d: dict) -> pyqtgraph.parametertree.Parameter:
    """Create a pyqtgraph Parameter from a dictionary.

    This function constructs a parameter tree that contains the settings defined in the
    provided dictionary. The structure of the dictionary should match the expected
    format for the ImageTool options.
    """
    return pyqtgraph.parametertree.Parameter.create(
        name="Settings",
        children=[
            {
                "name": "colors",
                "title": "Visualization",
                "type": "group",
                "children": [
                    {
                        "name": "cmap",
                        "title": "Colormap",
                        "type": "group",
                        "children": [
                            {
                                "name": "name",
                                "title": "Name",
                                "type": "erlabpy_colormap",
                                "value": d["colors"]["cmap"]["name"],
                                "default": DEFAULT_OPTIONS["colors"]["cmap"]["name"],
                            },
                            {
                                "name": "gamma",
                                "title": "Default γ",
                                "type": "float",
                                "value": d["colors"]["cmap"]["gamma"],
                                "default": DEFAULT_OPTIONS["colors"]["cmap"]["gamma"],
                                "limits": (0.01, 99.99),
                                "step": 0.01,
                            },
                            {
                                "name": "reverse",
                                "title": "Reverse",
                                "type": "bool",
                                "value": d["colors"]["cmap"]["reverse"],
                                "default": DEFAULT_OPTIONS["colors"]["cmap"]["reverse"],
                            },
                            {
                                "name": "exclude",
                                "title": "Exclude",
                                "type": "str",
                                "value": ", ".join(d["colors"]["cmap"]["exclude"]),
                                "default": ", ".join(
                                    DEFAULT_OPTIONS["colors"]["cmap"]["exclude"]
                                ),
                                "tip": "Comma-separated list of colormaps to hide from "
                                "the colormap selector.\n\nThis is useful to hide "
                                "colormaps that are very unlikely to be used.",
                            },
                            {
                                "name": "packages",
                                "title": "Packages",
                                "type": "str",
                                "value": ", ".join(d["colors"]["cmap"]["packages"]),
                                "default": ", ".join(
                                    DEFAULT_OPTIONS["colors"]["cmap"]["packages"]
                                ),
                                "tip": "Comma-separated list of additional packages "
                                "that provide colormaps.\n\nThe packages listed here "
                                "are not included in the default installation, but can "
                                "be installed separately.\nThey are loaded when the "
                                'user selects "Load All Colormaps" from the context '
                                "menu of the colormap selector.",
                            },
                        ],
                    },
                    {
                        "name": "cursors",
                        "title": "Cursor colors",
                        "type": "colorlist",
                        "value": d["colors"]["cursors"],
                        "default": DEFAULT_OPTIONS["colors"]["cursors"],
                        "tip": "Base colors for different cursors in the ImageTool.",
                    },
                ],
            },
            {
                "name": "io",
                "title": "I/O",
                "type": "group",
                "children": [
                    {
                        "name": "default_loader",
                        "title": "Default Loader",
                        "type": "list",
                        "value": d["io"]["default_loader"],
                        "default": DEFAULT_OPTIONS["io"]["default_loader"],
                        "limits": [None, *list(erlab.io.loaders.keys())],
                        "tip": "Default loader to use in the data explorer.",
                    },
                ],
            },
        ],
    )


def parameter_to_dict(param: pyqtgraph.parametertree.Parameter) -> dict:
    """Convert a pyqtgraph Parameter to a dictionary.

    This function extracts the values from the parameter tree and returns them in a
    dictionary format that has the same structure as `DEFAULT_OPTIONS`. The pyqtgraph
    Parameter is the one created by `make_parameter`.
    """
    color = param.child("colors")
    io = param.child("io")
    return {
        "colors": {
            "cmap": {
                "name": color.child("cmap").child("name").value(),
                "gamma": _as_float(color.child("cmap").child("gamma").value()),
                "reverse": _as_bool(color.child("cmap").child("reverse").value()),
                "exclude": [
                    s.strip()
                    for s in color.child("cmap").child("exclude").value().split(",")
                ],
                "packages": [
                    s.strip()
                    for s in color.child("cmap").child("packages").value().split(",")
                ],
            },
            "cursors": color.child("cursors").value(),
        },
        "io": {"default_loader": io.child("default_loader").value()},
    }


def _as_bool(value: typing.Any) -> bool:
    """Convert a value to a boolean."""
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _as_float(value: typing.Any) -> float:
    """Convert a value to a float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")
