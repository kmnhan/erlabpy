"""Whole-recipe Matplotlib rendering for Figure Composer."""

from __future__ import annotations

import contextlib
import typing

import matplotlib as mpl
import matplotlib.axes
import numpy as np
import xarray as xr

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._axes import _axes_expression_value
from erlab.interactive._figurecomposer._defaults import _figure_style_context
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_region_valid,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._sources import _valid_source_variable

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._state import (
        FigureAxesSelectionState,
        FigureGridSpecGridState,
    )
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _setup_kwargs(tool: FigureComposerTool) -> dict[str, typing.Any]:
    setup = tool._recipe.setup
    kwargs: dict[str, typing.Any] = {
        "nrows": setup.nrows,
        "ncols": setup.ncols,
        "figsize": setup.figsize,
        "dpi": setup.dpi,
        "squeeze": False,
    }
    if setup.layout is not None:
        kwargs["layout"] = setup.layout
    if setup.sharex is not False:
        kwargs["sharex"] = setup.sharex
    if setup.sharey is not False:
        kwargs["sharey"] = setup.sharey
    if setup.width_ratios and len(setup.width_ratios) == setup.ncols:
        kwargs["width_ratios"] = setup.width_ratios
    if setup.height_ratios and len(setup.height_ratios) == setup.nrows:
        kwargs["height_ratios"] = setup.height_ratios
    return kwargs


def _setup_layout_value(
    tool: FigureComposerTool,
) -> typing.Literal["constrained", "compressed", "tight", "none"] | None:
    return tool._recipe.setup.layout


def _set_creation_layout_engine(
    figure: Figure,
    layout: typing.Literal["constrained", "compressed", "tight", "none"] | None,
) -> None:
    if layout is None:
        figure.set_layout_engine(None)
        return
    if layout == "none":
        # Reused Figure instances can already have a layout engine. Matplotlib's
        # set_layout_engine("none") keeps a placeholder engine in that case,
        # while Figure(..., layout="none") starts with no engine. Match the
        # creation-time semantics used by generated code.
        with mpl.rc_context(
            {
                "figure.autolayout": False,
                "figure.constrained_layout.use": False,
            }
        ):
            figure.set_layout_engine(None)
        return
    figure.set_layout_engine(layout)


def _make_axes(
    tool: FigureComposerTool,
    figure: Figure | None = None,
    *,
    sync_visible: bool = True,
) -> np.ndarray | dict[str, matplotlib.axes.Axes]:
    if (
        sync_visible
        and tool._figure_window is not None
        and erlab.interactive.utils.qt_is_valid(tool._figure_window)
        and tool._figure_window.isVisible()
    ):
        tool._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
    if figure is None:
        figure = tool.figure
    figure.clear()
    setup = tool._recipe.setup
    figure.set_facecolor(mpl.rcParams["figure.facecolor"])
    figure.set_edgecolor(mpl.rcParams["figure.edgecolor"])
    figure.set_size_inches(setup.figsize, forward=False)
    figure_any = typing.cast("typing.Any", figure)
    if getattr(figure_any, "_original_dpi", setup.dpi) != setup.dpi:
        figure_any._original_dpi = setup.dpi
        figure_any._set_dpi(
            setup.dpi * getattr(figure_any.canvas, "device_pixel_ratio", 1),
            forward=False,
        )
    with contextlib.suppress(ValueError, NotImplementedError):
        _set_creation_layout_engine(figure, _setup_layout_value(tool))
    if setup.layout_mode == "gridspec":
        return _make_gridspec_axes(tool, figure)
    return np.asarray(
        figure.subplots(
            setup.nrows,
            setup.ncols,
            sharex=setup.sharex,
            sharey=setup.sharey,
            width_ratios=setup.width_ratios
            if len(setup.width_ratios) == setup.ncols
            else None,
            height_ratios=setup.height_ratios
            if len(setup.height_ratios) == setup.nrows
            else None,
            squeeze=False,
        ),
        dtype=object,
    )


def _make_gridspec_axes(
    tool: FigureComposerTool, figure: Figure
) -> dict[str, matplotlib.axes.Axes]:
    setup = tool._recipe.setup
    axes_by_id: dict[str, matplotlib.axes.Axes] = {}

    def gridspec_kwargs(grid: FigureGridSpecGridState) -> dict[str, typing.Any]:
        kwargs: dict[str, typing.Any] = {
            "nrows": grid.nrows,
            "ncols": grid.ncols,
        }
        if grid.width_ratios and len(grid.width_ratios) == grid.ncols:
            kwargs["width_ratios"] = grid.width_ratios
        if grid.height_ratios and len(grid.height_ratios) == grid.nrows:
            kwargs["height_ratios"] = grid.height_ratios
        if grid.wspace is not None:
            kwargs["wspace"] = grid.wspace
        if grid.hspace is not None:
            kwargs["hspace"] = grid.hspace
        return kwargs

    def build_grid(grid: FigureGridSpecGridState, spec: typing.Any) -> None:
        for axis_state in grid.axes:
            if _gridspec_region_valid(grid, axis_state.span):
                axes_by_id[axis_state.axes_id] = figure.add_subplot(
                    spec[
                        axis_state.span.row_start : axis_state.span.row_stop,
                        axis_state.span.col_start : axis_state.span.col_stop,
                    ]
                )
        for child in grid.child_grids:
            if child.span is None or not _gridspec_region_valid(grid, child.span):
                continue
            child_spec = spec[
                child.span.row_start : child.span.row_stop,
                child.span.col_start : child.span.col_stop,
            ].subgridspec(**gridspec_kwargs(child))
            build_grid(child, child_spec)

    root = setup.gridspec.root
    root_spec = figure.add_gridspec(**gridspec_kwargs(root))
    build_grid(root, root_spec)
    return axes_by_id


def _source_namespace(
    tool: FigureComposerTool,
    fig: Figure,
    axs: np.ndarray | dict[str, matplotlib.axes.Axes],
) -> dict[str, typing.Any]:
    first_axis = next(iter(axs.values()), None) if isinstance(axs, dict) else axs[0, 0]
    namespace: dict[str, typing.Any] = {
        "fig": fig,
        "axs": axs,
        "ax": first_axis,
        "np": np,
        "xr": xr,
        "eplt": eplt,
    }
    for name, data in tool._source_data.items():
        namespace[_valid_source_variable(name)] = data
    return namespace


def _axes_from_selection(
    tool: FigureComposerTool,
    selection: FigureAxesSelectionState,
    axs: np.ndarray | dict[str, matplotlib.axes.Axes],
    *,
    for_plot_slices: bool,
) -> object:
    if selection.expression:
        if isinstance(axs, dict):
            raise ValueError("Advanced axes expressions are only supported by subplots")
        return _axes_expression_value(selection.expression, axs)

    if isinstance(axs, dict):
        valid_ids = _gridspec_valid_axes_ids(tool._recipe.setup, selection.axes_ids)
        invalid_ids = tuple(
            axis_id for axis_id in selection.axes_ids if axis_id not in valid_ids
        )
        if invalid_ids:
            raise ValueError("Selected axes are outside the current GridSpec layout")
        selected = [axs[axis_id] for axis_id in valid_ids if axis_id in axs]
        if not selected:
            raise ValueError("No axes are selected for this operation")
        if for_plot_slices:
            return np.asarray(selected, dtype=object)
        if len(selected) == 1:
            return selected[0]
        return np.asarray(selected, dtype=object)

    if selection.invalid_axes(tool._recipe.setup):
        raise ValueError("Selected axes are outside the current figure layout")
    selected = [axs[row, col] for row, col in selection.valid_axes(tool._recipe.setup)]
    if not selected:
        raise ValueError("No axes are selected for this operation")
    if for_plot_slices:
        return np.asarray(selected, dtype=object)
    if len(selected) == 1:
        return selected[0]
    return np.asarray(selected, dtype=object)


def _iter_axes(axis_obj: object) -> tuple[matplotlib.axes.Axes, ...]:
    if isinstance(axis_obj, dict):
        return tuple(axis_obj.values())
    if isinstance(axis_obj, np.ndarray):
        return tuple(axis_obj.flat)
    if isinstance(axis_obj, (list, tuple)):
        return tuple(typing.cast("tuple[matplotlib.axes.Axes, ...]", axis_obj))
    return (typing.cast("matplotlib.axes.Axes", axis_obj),)


def _render_into_figure(
    tool: FigureComposerTool, figure: Figure, *, sync_visible: bool
) -> None:
    from erlab.interactive._figurecomposer._operations import _registry

    render_errors: dict[str, str] = {}
    with _figure_style_context():
        axs = _make_axes(tool, figure, sync_visible=sync_visible)
        for operation in tool._recipe.operations:
            if not operation.enabled:
                continue
            spec = _registry.spec_for(operation.kind)
            if spec.has_invalid_target(tool, operation):
                continue
            try:
                spec.render(tool, operation, figure, axs)
            except Exception as exc:
                render_errors[operation.operation_id] = _render_error_text(exc)
    tool._set_operation_render_errors(render_errors)


def _render_error_text(error: Exception) -> str:
    detail = str(error)
    if detail:
        return f"{type(error).__name__}: {detail}"
    return type(error).__name__


def _render_preview(
    tool: FigureComposerTool, *, show_window: bool | None = None
) -> None:
    if tool._rendering:
        return
    tool._rendering = True
    try:
        if show_window is None:
            show_window = (
                tool._figure_window is not None
                and erlab.interactive.utils.qt_is_valid(tool._figure_window)
                and tool._figure_window.isVisible()
            )
        if show_window:
            tool.canvas.flush_events()
            tool._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
        _render_into_figure(tool, tool.figure, sync_visible=True)
        if show_window:
            tool.canvas.draw()
            tool.canvas.flush_events()
        elif (
            tool._figure_window is not None
            and erlab.interactive.utils.qt_is_valid(tool._figure_window)
            and tool._figure_window.isVisible()
        ):
            tool.canvas.draw_idle()
    finally:
        tool._rendering = False
