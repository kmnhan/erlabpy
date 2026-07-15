"""Render plot-slices operations into Matplotlib figures."""

from __future__ import annotations

import math
import typing

import numpy as np

import erlab.plotting as eplt
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _normalized_selection_operation,
    _operation_maps,
    _plot_slices_kwargs,
    _plot_slices_panel_keys,
    _plot_slices_selection_sources,
    _plot_slices_slice_count,
    _plot_slices_transformed_kwargs,
    _plot_slices_transformed_maps,
    _plot_slices_uses_transformed_line_maps,
    _PlotSlicesPanelKey,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import matplotlib.axes
    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import FigureOperationState
    from erlab.interactive._figurecomposer._tool import FigureComposerTool

_PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR = "_figure_composer_operation_id"

_PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR = "_figure_composer_panel_key"


def _render_plot_slices(
    tool: FigureComposerTool, operation: FigureOperationState, axs: typing.Any
) -> None:
    operation = _normalized_selection_operation(tool._document, operation)
    maps = _operation_maps(tool._document, operation)
    if not maps:
        return
    kwargs = _plot_slices_kwargs(tool, operation)
    if _plot_slices_uses_transformed_line_maps(tool, operation):
        maps = _plot_slices_transformed_maps(tool, operation, maps)
        kwargs = _plot_slices_transformed_kwargs(tool, operation)
    selection_cache = getattr(tool, "_plot_slices_selection_cache", None)
    if selection_cache is not None:
        kwargs["_selection_cache"] = selection_cache
        kwargs["_selection_cache_key"] = _plot_slices_selection_cache_key(
            operation, maps
        )
    axes = _plot_slices_axes(
        operation,
        maps,
        _axes_from_selection(tool, operation.axes, axs, for_plot_slices=True),
        slice_count=_plot_slices_slice_count(tool._document, operation),
    )
    axes_tuple = _iter_axes(axes)
    panel_keys = _plot_slices_panel_keys(
        tool._document, tool._source_display_name, operation
    )
    mappable_ids_before = _axis_mappable_ids(axes_tuple)
    eplt.plot_slices(
        maps,
        axes=typing.cast("Iterable[matplotlib.axes.Axes]", axes),
        **kwargs,
    )
    _tag_plot_slices_mappables(
        operation,
        axes_tuple,
        panel_keys,
        mappable_ids_before,
    )


def _axis_mappables(axis: object) -> tuple[object, ...]:
    images = tuple(getattr(axis, "images", ()))
    collections = tuple(getattr(axis, "collections", ()))
    return (*images, *collections)


def _axis_mappable_ids(axes: Sequence[object]) -> dict[object, set[int]]:
    return {axis: {id(mappable) for mappable in _axis_mappables(axis)} for axis in axes}


def _tag_plot_slices_mappables(
    operation: FigureOperationState,
    axes: Sequence[object],
    panel_keys: Sequence[_PlotSlicesPanelKey],
    mappable_ids_before: Mapping[object, set[int]],
) -> None:
    if len(axes) != len(panel_keys):
        return
    for axis, panel_key in zip(axes, panel_keys, strict=True):
        previous_ids = mappable_ids_before.get(axis, set())
        for mappable in _axis_mappables(axis):
            if id(mappable) in previous_ids:
                continue
            setattr(
                mappable,
                _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
                operation.operation_id,
            )
            setattr(
                mappable,
                _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
                (panel_key.map_index, panel_key.slice_index),
            )


def _plot_slices_axes(
    operation: FigureOperationState,
    maps: Sequence[xr.DataArray],
    axes: object,
    *,
    slice_count: int | None = None,
) -> object:
    if not isinstance(axes, np.ndarray):
        return axes
    slice_count = max(
        slice_count if slice_count is not None else len(operation.slice_values), 1
    )
    if operation.order == "F":
        shape = (slice_count, len(maps))
    else:
        shape = (len(maps), slice_count)
    if axes.size != math.prod(shape):
        return axes
    return axes.reshape(shape)


def _plot_slices_selection_cache_key(
    operation: FigureOperationState, maps: Sequence[xr.DataArray]
) -> tuple[object, ...]:
    source_key = tuple(
        (source,) for source in _plot_slices_selection_sources(operation)
    )
    map_key = tuple(
        (id(data.data), tuple(data.dims), tuple(data.shape)) for data in maps
    )
    return (source_key, map_key)
