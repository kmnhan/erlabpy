"""Persistent Figure Composer recipe state."""

from __future__ import annotations

import enum
import typing
import uuid

import pydantic

from erlab.interactive._figurecomposer._defaults import (
    _default_export_bbox_inches,
    _default_export_dpi,
    _default_export_transparent,
    _default_figsize,
    _default_figure_dpi,
    _default_layout,
)
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class FigureSourceState(pydantic.BaseModel):
    """A named data source used by a figure recipe."""

    name: str
    label: str
    node_uid: str | None = None
    node_snapshot_token: str | None = None
    provenance_spec: dict[str, typing.Any] | None = None

    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def from_script_input(
        cls, script_input: provenance.ScriptInput
    ) -> FigureSourceState:
        return cls(
            name=script_input.name,
            label=script_input.label,
            node_uid=script_input.node_uid,
            node_snapshot_token=script_input.node_snapshot_token,
            provenance_spec=script_input.provenance_spec,
        )

    def to_script_input(self) -> provenance.ScriptInput | None:
        if self.node_uid is None and self.provenance_spec is None:
            return None
        return provenance.ScriptInput(
            name=self.name,
            label=self.label,
            node_uid=self.node_uid,
            node_snapshot_token=self.node_snapshot_token,
            provenance_spec=self.provenance_spec,
        )


class FigureSubplotsState(pydantic.BaseModel):
    """Persistent ``plt.subplots`` setup for a figure recipe."""

    nrows: int = 1
    ncols: int = 1
    figsize: tuple[float, float] = pydantic.Field(default_factory=_default_figsize)
    dpi: float = pydantic.Field(default_factory=_default_figure_dpi)
    layout: typing.Literal["constrained", "compressed", "tight"] | None = (
        pydantic.Field(default_factory=_default_layout)
    )
    sharex: bool | typing.Literal["none", "all", "row", "col"] = "col"
    sharey: bool | typing.Literal["none", "all", "row", "col"] = "row"
    width_ratios: tuple[float, ...] = ()
    height_ratios: tuple[float, ...] = ()

    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("nrows", "ncols")
    @classmethod
    def _validate_grid_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("subplot grids must have at least one row and column")
        return value

    @pydantic.field_validator("figsize")
    @classmethod
    def _validate_figsize(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("figsize values must be positive")
        return value

    @pydantic.field_validator("dpi")
    @classmethod
    def _validate_dpi(cls, value: float) -> float:
        if value < 1:
            raise ValueError("dpi must be positive")
        return value

    @pydantic.field_validator("width_ratios", "height_ratios")
    @classmethod
    def _validate_ratios(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item <= 0.0 for item in value):
            raise ValueError("subplot ratios must be positive")
        return value


class FigureAxesSelectionState(pydantic.BaseModel):
    """Stable 2D ``axs[row, col]`` target selection."""

    axes: tuple[tuple[int, int], ...] = ((0, 0),)
    expression: str = ""

    model_config = pydantic.ConfigDict(extra="forbid")

    def valid_axes(self, setup: FigureSubplotsState) -> tuple[tuple[int, int], ...]:
        return tuple(
            dict.fromkeys(
                (row, col)
                for row, col in self.axes
                if 0 <= row < setup.nrows and 0 <= col < setup.ncols
            )
        )

    def invalid_axes(self, setup: FigureSubplotsState) -> tuple[tuple[int, int], ...]:
        return tuple(
            dict.fromkeys(
                (row, col)
                for row, col in self.axes
                if row < 0 or row >= setup.nrows or col < 0 or col >= setup.ncols
            )
        )

    def bounded(self, setup: FigureSubplotsState) -> FigureAxesSelectionState:
        axes = list(self.valid_axes(setup))
        if not axes:
            axes.append((0, 0))
        return self.model_copy(update={"axes": tuple(axes)})


class FigureExportState(pydantic.BaseModel):
    """Default export settings for the composer."""

    dpi: float | typing.Literal["figure"] = pydantic.Field(
        default_factory=_default_export_dpi
    )
    transparent: bool = pydantic.Field(default_factory=_default_export_transparent)
    bbox_inches: str | None = pydantic.Field(
        default_factory=_default_export_bbox_inches
    )

    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("dpi", mode="before")
    @classmethod
    def _validate_export_dpi(
        cls, value: typing.Any
    ) -> float | typing.Literal["figure"]:
        if value == "figure":
            return "figure"
        return float(value)


class FigureOperationKind(enum.StrEnum):
    PLOT_SLICES = "plot_slices"
    LINE = "line"
    METHOD = "method"
    CUSTOM = "custom"


class FigureMethodFamily(enum.StrEnum):
    AXES = "axes"
    FIGURE = "figure"
    ERLAB = "erlab"


_POWER_NORM_NAME = "PowerNorm"
FigureLimit = float | tuple[float, float]


class _PlotSlicesShape(typing.NamedTuple):
    source_text: str
    selection_text: str
    panel_text: str
    axes_text: str
    plot_dims: tuple[str, ...]
    plot_ndim: int | None
    panel_count: int
    valid: bool


class FigureDataSelectionState(pydantic.BaseModel):
    """Structured source selection used by plot_slices and line operations."""

    source: str
    isel: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    qsel: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    mean_dims: tuple[str, ...] = ()

    model_config = pydantic.ConfigDict(extra="forbid")


class FigureOperationState(pydantic.BaseModel):
    """One ordered plotting operation in a figure recipe."""

    operation_id: str = pydantic.Field(default_factory=lambda: uuid.uuid4().hex)
    kind: FigureOperationKind
    label: str
    enabled: bool = True
    axes: FigureAxesSelectionState = pydantic.Field(
        default_factory=FigureAxesSelectionState
    )

    sources: tuple[str, ...] = ()
    map_selections: tuple[FigureDataSelectionState, ...] = ()
    slice_dim: str | None = None
    slice_values: tuple[float, ...] = ()
    slice_width: float | None = None
    transpose: bool = False
    xlim: FigureLimit | None = None
    ylim: FigureLimit | None = None
    crop: bool = True
    same_limits: bool | str = False
    axis: str = "auto"
    show_all_labels: bool = False
    colorbar: str = "none"
    hide_colorbar_ticks: bool = True
    annotate: bool = True
    cmap: str | None = None
    gamma: float | None = None
    norm_name: str | None = None
    norm_gamma: float | None = None
    norm_clip: bool | None = None
    norm_kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    vmin: float | None = None
    vmax: float | None = None
    vcenter: float | None = None
    halfrange: float | None = None
    order: str = "C"
    cmap_order: str = "C"
    norm_order: str | None = None
    gradient: bool = False
    gradient_kw: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    subplot_kw: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    annotate_kw: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    colorbar_kw: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    extra_kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    line_source: str | None = None
    line_labels: tuple[str, ...] = ()
    line_color: str | None = None
    line_colors: tuple[str, ...] = ()
    line_x: str | None = None
    line_y: str | None = None
    line_selection: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    line_iter_dim: str | None = None
    line_normalize: typing.Literal["none", "max", "mean"] = "none"
    line_placement: typing.Literal["all_axes", "one_per_axis"] = "all_axes"
    line_values_axis: typing.Literal["x", "y"] = "y"
    line_scales: tuple[float, ...] = ()
    line_offsets: tuple[float, ...] = ()
    line_offset_source: typing.Literal[
        "manual", "index", "coordinate", "associated"
    ] = "manual"
    line_offset_coord: str | None = None
    line_offset_scale: float = 1.0

    method_family: FigureMethodFamily = FigureMethodFamily.ERLAB
    method_name: str = "clean_labels"
    method_args: tuple[typing.Any, ...] = ()
    method_kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    method_call_policy: str | None = None
    text_values: tuple[str, ...] = ()
    method_coordinate_system: typing.Literal["data", "axes"] = "data"

    code: str = ""
    trusted: bool = False

    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def plot_slices(
        cls,
        *,
        label: str,
        sources: Sequence[str],
        map_selections: Sequence[FigureDataSelectionState] = (),
        axes: FigureAxesSelectionState | None = None,
        slice_dim: str | None = None,
        slice_values: Sequence[float] = (),
        slice_width: float | None = None,
    ) -> FigureOperationState:
        return cls(
            kind=FigureOperationKind.PLOT_SLICES,
            label=label,
            sources=tuple(sources),
            map_selections=tuple(map_selections),
            axes=axes or FigureAxesSelectionState(),
            slice_dim=slice_dim,
            slice_values=tuple(slice_values),
            slice_width=slice_width,
            norm_name=_POWER_NORM_NAME,
        )

    @classmethod
    def line(
        cls,
        *,
        label: str,
        source: str,
        axes: FigureAxesSelectionState | None = None,
    ) -> FigureOperationState:
        return cls(
            kind=FigureOperationKind.LINE,
            label=label,
            line_source=source,
            axes=axes or FigureAxesSelectionState(),
        )

    @classmethod
    def method(
        cls,
        *,
        family: FigureMethodFamily,
        name: str,
        label: str | None = None,
        axes: FigureAxesSelectionState | None = None,
        args: Sequence[typing.Any] = (),
        kwargs: Mapping[str, typing.Any] | None = None,
    ) -> FigureOperationState:
        return cls(
            kind=FigureOperationKind.METHOD,
            label=label or name,
            method_family=family,
            method_name=name,
            method_args=tuple(args),
            method_kwargs=dict(kwargs or {}),
            axes=axes or FigureAxesSelectionState(),
        )

    @classmethod
    def custom(cls, *, label: str, code: str, trusted: bool) -> FigureOperationState:
        return cls(
            kind=FigureOperationKind.CUSTOM,
            label=label,
            code=code,
            trusted=trusted,
            axes=FigureAxesSelectionState(axes=()),
        )


class FigureRecipeState(pydantic.BaseModel):
    """Complete persistent recipe for a Figure Composer window."""

    setup: FigureSubplotsState = pydantic.Field(default_factory=FigureSubplotsState)
    sources: tuple[FigureSourceState, ...] = ()
    operations: tuple[FigureOperationState, ...] = ()
    export: FigureExportState = pydantic.Field(default_factory=FigureExportState)
    primary_source: str = "data"

    model_config = pydantic.ConfigDict(extra="forbid")
