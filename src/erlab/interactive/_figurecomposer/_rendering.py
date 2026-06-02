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
from erlab.interactive._figurecomposer._sources import _valid_source_variable

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._state import FigureAxesSelectionState
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
    return kwargs


def _make_axes(
    tool: FigureComposerTool,
    figure: Figure | None = None,
    *,
    sync_visible: bool = True,
) -> np.ndarray:
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
        figure.set_layout_engine(setup.layout)
    return np.asarray(
        figure.subplots(
            setup.nrows,
            setup.ncols,
            sharex=setup.sharex,
            sharey=setup.sharey,
            squeeze=False,
        ),
        dtype=object,
    )


def _source_namespace(
    tool: FigureComposerTool, fig: Figure, axs: np.ndarray
) -> dict[str, typing.Any]:
    namespace: dict[str, typing.Any] = {
        "fig": fig,
        "axs": axs,
        "ax": axs[0, 0],
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
    axs: np.ndarray,
    *,
    for_plot_slices: bool,
) -> object:
    if selection.expression:
        return _axes_expression_value(selection.expression, axs)

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
    if isinstance(axis_obj, np.ndarray):
        return tuple(axis_obj.flat)
    if isinstance(axis_obj, (list, tuple)):
        return tuple(typing.cast("tuple[matplotlib.axes.Axes, ...]", axis_obj))
    return (typing.cast("matplotlib.axes.Axes", axis_obj),)


def _render_into_figure(
    tool: FigureComposerTool, figure: Figure, *, sync_visible: bool
) -> None:
    from erlab.interactive._figurecomposer._operations import _registry

    with _figure_style_context():
        axs = _make_axes(tool, figure, sync_visible=sync_visible)
        for operation in tool._recipe.operations:
            if not operation.enabled:
                continue
            spec = _registry.spec_for(operation.kind)
            if spec.has_invalid_target(tool, operation):
                continue
            with contextlib.suppress(Exception):
                spec.render(tool, operation, figure, axs)


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
