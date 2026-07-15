"""Shared generated-code helpers for Figure Composer operations."""

from __future__ import annotations

import typing

import erlab
from erlab.interactive._figurecomposer._model._axes import (
    _compact_axes_code,
    _compact_axes_iterable_code,
)
from erlab.interactive._figurecomposer._model._gridspec import (
    _gridspec_axis_code_names,
    _gridspec_axis_code_tuple,
    _gridspec_has_invalid_regions,
    _gridspec_invalid_axes_ids,
    _gridspec_span_code,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._model._sources import (
    _decode_indexers,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._rendering import (
    _setup_kwargs,
    _setup_layout_value,
)
from erlab.interactive._figurecomposer._text import _code_kwargs, _format_axes_tuple

if typing.TYPE_CHECKING:
    import xarray as xr

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._model._state import (
        FigureAxesSelectionState,
        FigureDataSelectionState,
        FigureGridSpecGridState,
    )


def _invalid_gridspec_axes_error(invalid_axes: tuple[str, ...]) -> str:
    count = len(invalid_axes)
    suffix = "axis" if count == 1 else "axes"
    return f"{count} selected GridSpec {suffix} outside the current layout"


def _reserved_code_names(context: FigureRecipeContext) -> tuple[str, ...]:
    return tuple(context.source_names())


def _axes_code(
    context: FigureRecipeContext,
    selection: FigureAxesSelectionState,
    *,
    for_plot_slices: bool,
) -> str:
    if selection.expression:
        return selection.expression

    setup = context.recipe.setup
    if setup.layout_mode == "gridspec":
        invalid_axes = _gridspec_invalid_axes_ids(setup, selection.axes_ids)
        if invalid_axes:
            raise ValueError(_invalid_gridspec_axes_error(invalid_axes))
        axes_ids = _gridspec_valid_axes_ids(setup, selection.axes_ids)
        if not axes_ids:
            raise ValueError("No axes are selected")
        axes_code = _gridspec_axis_code_tuple(
            setup, axes_ids, reserved_names=_reserved_code_names(context)
        )
        if for_plot_slices:
            return "[" + ", ".join(axes_code) + "]"
        if len(axes_code) == 1:
            return axes_code[0]
        return "[" + ", ".join(axes_code) + "]"

    invalid_axes = selection.invalid_axes(context.recipe.setup)
    if invalid_axes:
        raise ValueError(
            f"Selected axes are outside the current layout: "
            f"{_format_axes_tuple(invalid_axes)}"
        )
    axes = selection.valid_axes(context.recipe.setup)
    if not axes:
        raise ValueError("No axes are selected")
    compact = _compact_axes_code(axes, nrows=setup.nrows, ncols=setup.ncols)
    if compact is not None:
        if for_plot_slices and len(axes) == 1:
            return f"[{compact}]"
        return compact

    items = ", ".join(f"axs[{row}, {col}]" for row, col in axes)
    return f"[{items}]"


def _axes_sequence_code(
    context: FigureRecipeContext, selection: FigureAxesSelectionState
) -> str:
    if selection.expression:
        return _axes_code(context, selection, for_plot_slices=True)
    setup = context.recipe.setup
    if setup.layout_mode == "gridspec":
        invalid_axes = _gridspec_invalid_axes_ids(setup, selection.axes_ids)
        if invalid_axes:
            raise ValueError(_invalid_gridspec_axes_error(invalid_axes))
        axes_ids = _gridspec_valid_axes_ids(setup, selection.axes_ids)
        if not axes_ids:
            raise ValueError("No axes are selected")
        axes_code = _gridspec_axis_code_tuple(
            setup, axes_ids, reserved_names=_reserved_code_names(context)
        )
        if len(axes_code) == 1:
            return f"({axes_code[0]},)"
        return "(" + ", ".join(axes_code) + ")"

    invalid_axes = selection.invalid_axes(context.recipe.setup)
    if invalid_axes:
        raise ValueError(
            f"Selected axes are outside the current layout: "
            f"{_format_axes_tuple(invalid_axes)}"
        )
    axes = selection.valid_axes(context.recipe.setup)
    if not axes:
        raise ValueError("No axes are selected")
    code = _compact_axes_iterable_code(axes, nrows=setup.nrows, ncols=setup.ncols)
    if code is None:
        raise ValueError("No axes are selected")
    return code


def _selection_code(selection: FigureDataSelectionState) -> str:
    code = _valid_source_variable(selection.source)
    if selection.isel:
        code += f".isel({_code_kwargs(_decode_indexers(selection.isel))})"
    if selection.qsel:
        code += f".qsel({_code_kwargs(selection.qsel)})"
    if selection.mean_dims:
        if len(selection.mean_dims) == 1:
            mean_arg = erlab.interactive.utils._parse_single_arg(selection.mean_dims[0])
        else:
            mean_arg = erlab.interactive.utils._parse_single_arg(selection.mean_dims)
        code += f".qsel.mean({mean_arg})"
    return code


def _needs_squeeze_drop(data: xr.DataArray) -> bool:
    return any(size == 1 for size in data.sizes.values())


def _maybe_squeeze_drop_code(code: str, data: xr.DataArray) -> str:
    if _needs_squeeze_drop(data):
        return f"{code}.squeeze(drop=True)"
    return code


def _setup_code(context: FigureRecipeContext) -> str:
    setup = context.recipe.setup
    if setup.layout_mode == "gridspec":
        if _gridspec_has_invalid_regions(setup.gridspec.root):
            raise ValueError("GridSpec layout contains regions outside their grids")
        return "\n".join(_gridspec_setup_code_lines(context))
    kwargs = _code_kwargs(_setup_kwargs(context))
    return f"fig, axs = plt.subplots({kwargs})"


def _gridspec_setup_code_lines(context: FigureRecipeContext) -> list[str]:
    setup = context.recipe.setup
    kwargs = {
        "figsize": setup.figsize,
        "dpi": setup.dpi,
    }
    if (layout := _setup_layout_value(context)) is not None:
        kwargs["layout"] = layout
    lines = [f"fig = plt.figure({_code_kwargs(kwargs)})"]
    code_names = _gridspec_axis_code_names(
        setup, reserved_names=_reserved_code_names(context)
    )

    def grid_kwargs(grid: FigureGridSpecGridState) -> dict[str, typing.Any]:
        grid_kwargs: dict[str, typing.Any] = {
            "nrows": grid.nrows,
            "ncols": grid.ncols,
        }
        if grid.width_ratios and len(grid.width_ratios) == grid.ncols:
            grid_kwargs["width_ratios"] = grid.width_ratios
        if grid.height_ratios and len(grid.height_ratios) == grid.nrows:
            grid_kwargs["height_ratios"] = grid.height_ratios
        if grid.wspace is not None:
            grid_kwargs["wspace"] = grid.wspace
        if grid.hspace is not None:
            grid_kwargs["hspace"] = grid.hspace
        return grid_kwargs

    def append_grid_lines(
        grid: FigureGridSpecGridState, grid_var: str, *, root: bool
    ) -> None:
        if root:
            lines.append(
                f"{grid_var} = fig.add_gridspec({_code_kwargs(grid_kwargs(grid))})"
            )
        lines.extend(
            f"{code_names[axis.axes_id]} = fig.add_subplot("
            f"{grid_var}{_gridspec_span_code(axis.span, grid)})"
            for axis in grid.axes
        )
        for child_index, child in enumerate(grid.child_grids):
            child_var = f"{grid_var}_{child_index}"
            if child.span is None:
                continue
            lines.append(
                f"{child_var} = {grid_var}{_gridspec_span_code(child.span, grid)}"
                f".subgridspec({_code_kwargs(grid_kwargs(child))})"
            )
            append_grid_lines(child, child_var, root=False)

    append_grid_lines(setup.gridspec.root, "gs0", root=True)
    return lines
