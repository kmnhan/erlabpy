"""Shared generated-code helpers for Figure Composer operations."""

from __future__ import annotations

import typing

import erlab
from erlab.interactive._figurecomposer import _rendering
from erlab.interactive._figurecomposer._axes import (
    _compact_axes_code,
    _compact_axes_iterable_code,
)
from erlab.interactive._figurecomposer._sources import (
    _decode_indexers,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._text import _code_kwargs, _format_axes_tuple

if typing.TYPE_CHECKING:
    from erlab.interactive._figurecomposer._state import (
        FigureAxesSelectionState,
        FigureDataSelectionState,
    )
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _axes_code(
    tool: FigureComposerTool,
    selection: FigureAxesSelectionState,
    *,
    for_plot_slices: bool,
) -> str:
    if selection.expression:
        return selection.expression

    invalid_axes = selection.invalid_axes(tool._recipe.setup)
    if invalid_axes:
        raise ValueError(
            f"Selected axes are outside the current layout: "
            f"{_format_axes_tuple(invalid_axes)}"
        )
    axes = selection.valid_axes(tool._recipe.setup)
    if not axes:
        raise ValueError("No axes are selected")
    setup = tool._recipe.setup
    compact = _compact_axes_code(axes, nrows=setup.nrows, ncols=setup.ncols)
    if compact is not None:
        if for_plot_slices and len(axes) == 1:
            return f"[{compact}]"
        return compact

    items = ", ".join(f"axs[{row}, {col}]" for row, col in axes)
    return f"[{items}]"


def _axes_sequence_code(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> str:
    if selection.expression:
        return _axes_code(tool, selection, for_plot_slices=True)
    invalid_axes = selection.invalid_axes(tool._recipe.setup)
    if invalid_axes:
        raise ValueError(
            f"Selected axes are outside the current layout: "
            f"{_format_axes_tuple(invalid_axes)}"
        )
    axes = selection.valid_axes(tool._recipe.setup)
    if not axes:
        raise ValueError("No axes are selected")
    setup = tool._recipe.setup
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


def _setup_code(tool: FigureComposerTool) -> str:
    kwargs = _code_kwargs(_rendering._setup_kwargs(tool))
    return f"fig, axs = plt.subplots({kwargs})"
