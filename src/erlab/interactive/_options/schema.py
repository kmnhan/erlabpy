"""Schema for interactive tool options.

The GUI options window are automatically generated from this schema. Whenever the

To facillitate this, some extra metadata is provided in the `json_schema_extra` of
pydantic fields.

The following trivial types have automatic GUI representations:

- bool: checkbox
- int and float: numeric input (spinbox)
- list: comma-separated text input

For spinboxes, limits provided with `ge`, `le` are used to set the minimum and maximum
values of the spinbox. `gt` and `lt` are not supported because validation would fail for
values exactly at the limit. If only one limit is provided, the other limit is set to
positive or negative infinity.

For comma-separated lists, a custom validator must be implemented to split the string
into a list of strings. See methods decorated with `@field_validator` below for
examples.

To bypass automatic detection of type or to provide a custom type, provide "ui_type" in
the `json_schema_extra` dictionary, which will be directly used as `type` when creating
the pyqtgraph Parameter tree.

For comboboxes, provide "ui_type": "list" and "ui_limits": [...] in `json_schema_extra`.
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


class IOOptions(BaseModel):
    """Input/output related options."""

    default_loader: str = Field(
        default="None",
        title="Default loader",
        description="Loader to pre-select in the data explorer.",
        json_schema_extra={
            "ui_type": "list",
            "ui_limits": ["None", *list(erlab.io.loaders.keys())],
        },
    )

    compute_threshold: int = Field(
        default=2048,
        title="Dask computation threshold (MB)",
        description=(
            "Threshold in megabytes for automatically loading dask arrays into memory "
            "when showing dask-backed data in ImageTool."
            "\n\nData smaller than this threshold will be automatically "
            "computed and loaded into memory to improve interactivity."
        ),
        ge=0,
        le=1000000,
    )

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
