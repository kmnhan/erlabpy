"""Schema for interactive tool options.

The GUI options window are automatically generated from this schema. Whenever the

To facillitate this, some extra metadata is provided in the `json_schema_extra` of
pydantic fields.

The following trivial types have automatic GUI representations:

- bool: checkbox
- int and float: numeric input (spinbox)
- list: comma-separated text input

For spinboxes, limits provided with `ge`, `le` are used to set the minimum and maximum
values of the spinbox. If only one limit is provided, the other limit is set to positive
or negative infinity.

.. note::

    `gt` and `lt` are not supported because validation would fail for values exactly at
    the limit.

For comma-separated lists, a custom validator must be implemented to split the string
into a list of strings. See methods decorated with `@field_validator` below for
examples.

To bypass automatic detection of type or to provide a custom type, provide "ui_type" in
the `json_schema_extra` dictionary, which will be directly used as `type` when creating
the pyqtgraph Parameter tree.

To pass additional options to the pyqtgraph Parameter tree, prefix the option name with
"ui_" in the `json_schema_extra` dictionary. For example, to set the step size of a
float parameter, you can do this:

.. code-block:: python

    my_value: float = Field(
        default=1.0,
        json_schema_extra={"ui_step": 0.1},
    )

See the `pyqtgraph Parameter Tree documentation
<https://pyqtgraph.readthedocs.io/en/latest/api_reference/parametertree/>`_ for all
available options.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field, field_validator

import erlab

__all__ = ["AppOptions", "ColorMapOptions", "ColorOptions", "IOOptions"]


def _unique_seq(seq: list[str]) -> list[str]:
    """Return sequence with order-preserving uniqueness."""
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class KToolBZOptions(BaseModel):
    default_a: float = Field(
        default=3.54,
        title="a",
        description="Default lattice constant a in Ångström.",
        ge=0.01,
        le=99.99,
        json_schema_extra={"ui_step": 0.01, "ui_suffix": "Å"},
    )
    default_b: float = Field(
        default=3.54,
        title="b",
        description="Default lattice constant b in Ångström.",
        ge=0.01,
        le=99.99,
        json_schema_extra={"ui_step": 0.01, "ui_suffix": "Å"},
    )
    default_c: float = Field(
        default=6.01,
        title="c",
        description="Default lattice constant c in Ångström.",
        ge=0.01,
        le=99.99,
        json_schema_extra={"ui_step": 0.01, "ui_suffix": "Å"},
    )
    default_ang: float = Field(
        default=120.0,
        title="γ",
        description="Default lattice angle γ in degrees.",
        ge=1.0,
        le=179.0,
        json_schema_extra={"ui_step": 1.0, "ui_suffix": "°"},
    )
    default_rot: float = Field(
        default=0.0,
        title="rot",
        description="Default rotation of the Brillouin zone in degrees.",
        ge=-360.0,
        le=360.0,
        json_schema_extra={"ui_step": 1.0, "ui_suffix": "°"},
    )


class KToolOptions(BaseModel):
    """Momentum conversion tool related options."""

    bz: KToolBZOptions = Field(
        default_factory=KToolBZOptions,
        title="Brillouin zone",
    )


class DaskOptions(BaseModel):
    """Dask-related options."""

    compute_threshold: int = Field(
        default=256,
        title="Compute threshold",
        description=(
            "Threshold in megabytes for automatically loading dask arrays into memory "
            "when showing dask-backed data in ImageTool."
            "\n\nData smaller than this threshold will be automatically "
            "computed and loaded into memory to improve interactivity."
        ),
        ge=0,
        le=1000000,
        json_schema_extra={"ui_step": 128, "ui_suffix": "MB"},
    )


class IOOptions(BaseModel):
    """Top-level grouping of I/O-related options."""

    default_loader: str = Field(
        default="None",
        title="Default loader",
        description="Loader to pre-select in the data explorer.",
        json_schema_extra={
            "ui_type": "list",
            "ui_limits": ["None", *list(erlab.io.loaders.keys())],
        },
    )

    dask: DaskOptions = Field(default_factory=DaskOptions, title="Dask")

    @field_validator("default_loader", mode="before")
    @classmethod
    def loader_exists(cls, v: str | None):
        if not v or v == "None":
            return "None"
        if v not in erlab.io.loaders:
            available = list(erlab.io.loaders.keys())
            raise ValueError(
                "Loader '" + v + "' not registered; available: " + str(available)
            )
        return v


class ColorMapOptions(BaseModel):
    """Colormap related visualization options."""

    name: str = Field(
        default="magma",
        title="Name",
        description="Name of the default colormap.",
        json_schema_extra={"ui_type": "erlabpy_colormap"},
    )
    gamma: float = Field(
        default=0.5,
        ge=0.01,
        le=99.99,
        title="Default γ",
        description="Default gamma exponent.",
        json_schema_extra={"ui_step": 0.01},
    )
    reverse: bool = Field(
        default=False,
        title="Reverse",
        description="Display the colormap reversed by default.",
    )
    exclude: list[str] = Field(
        default=[
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
        title="Exclude",
        description="Comma-separated list of colormaps to hide from the selector."
        "\n\nThis is useful to hide colormaps that are very unlikely to be used.",
    )
    packages: list[str] = Field(
        default=[
            "cmasher",
            "cmocean",
            "colorcet",
            "cmcrameri",
            "seaborn",
        ],
        title="Packages",
        description=(
            "Comma-separated list of additional packages providing extra colormaps."
            "\n\nImported on demand when the user selects 'Load All Colormaps' from "
            "the context menu of the colormap selector."
        ),
    )

    @field_validator("exclude", "packages", mode="before")
    @classmethod
    def split_str(cls, v: typing.Any):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        return _unique_seq(v)


class ColorOptions(BaseModel):
    """Top-level grouping of color-related options."""

    cmap: ColorMapOptions = Field(default_factory=ColorMapOptions, title="Colormap")
    cursors: list[str] = Field(
        default=[
            "#cccccc",
            "#ffff00",
            "#ff00ff",
            "#00ffff",
            "#008000",
            "#ff0000",
            "#0000ff",
        ],
        title="Cursor colors",
        description="Base list of colors used for different cursors in the ImageTool.",
        json_schema_extra={"ui_type": "colorlist"},
    )

    @field_validator("cursors", mode="before")
    @classmethod
    def split_cursors(cls, v: typing.Any):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        return v


class AppOptions(BaseModel):
    """Root configuration model for interactive tool options."""

    colors: ColorOptions = Field(
        default_factory=ColorOptions,
        title="Visualization",
        description="Visualization and colormap related settings.",
    )
    io: IOOptions = Field(
        default_factory=IOOptions,
        title="I/O",
        description="Input/output defaults.",
    )
    ktool: KToolOptions = Field(
        default_factory=KToolOptions,
        title="ktool",
        description="Momentum conversion tool defaults.",
    )
