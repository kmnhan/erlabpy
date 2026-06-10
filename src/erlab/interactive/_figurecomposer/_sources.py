"""Source and data-selection helpers for Figure Composer."""

from __future__ import annotations

import contextlib
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

from erlab.interactive._figurecomposer._axes import _all_axes
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
    FigureOperationState,
    FigureSourceState,
    FigureSubplotsState,
)


def _source_name(data: xr.DataArray) -> str:
    if isinstance(data.name, str) and data.name.isidentifier():
        return data.name
    return "data"


def _source_label(data: xr.DataArray) -> str:
    if data.name is None:
        return "data"
    return str(data.name)


def _source_display_label(
    source: FigureSourceState | None, name: str, *, disambiguate: bool = False
) -> str:
    """Return the user-facing source label for a recipe source alias."""
    label = name if source is None else source.label.strip() or name
    if source is not None and disambiguate and label != name:
        return f"{label} ({name})"
    return label


def _source_duplicate_labels(
    sources: Sequence[FigureSourceState],
) -> frozenset[str]:
    counts: dict[str, int] = {}
    for source in sources:
        label = source.label.strip() or source.name
        counts[label] = counts.get(label, 0) + 1
    return frozenset(label for label, count in counts.items() if count > 1)


def _source_display_tooltip(source: FigureSourceState | None, name: str) -> str:
    if source is None:
        return f"Alias: {name}"
    label = source.label.strip() or name
    if label == name:
        return f"Alias: {name}"
    return f"{label}\nAlias: {name}"


def _valid_source_variable(name: str) -> str:
    if not name.isidentifier():
        raise ValueError(f"Figure source name {name!r} is not a valid variable name")
    return name


def _available_source_dims(
    source_data: Mapping[str, xr.DataArray], sources: Sequence[str]
) -> list[str]:
    dims: list[str] = []
    for source in sources:
        data = source_data.get(source)
        if data is None:
            continue
        for dim in data.dims:
            if dim not in dims:
                dims.append(str(dim))
    return dims


def _selected_data(
    source_data: Mapping[str, xr.DataArray], selection: FigureDataSelectionState
) -> xr.DataArray | None:
    selected = source_data.get(selection.source)
    if selected is None:
        return None
    if selection.isel:
        selected = selected.isel(_decode_indexers(selection.isel))
    if selection.qsel:
        selected = selected.qsel(selection.qsel)
    if selection.mean_dims:
        mean_arg: str | tuple[str, ...]
        if len(selection.mean_dims) == 1:
            mean_arg = selection.mean_dims[0]
        else:
            mean_arg = selection.mean_dims
        selected = selected.qsel.mean(mean_arg)
    return selected


def _middle_coord_value(data: xr.DataArray, dim: str) -> float | None:
    coord = data.coords.get(dim)
    if coord is None or coord.size == 0:
        return None
    value = coord.values[int(coord.size // 2)]
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return None


def _default_plot_operation(
    name: str,
    data: xr.DataArray,
    *,
    setup: FigureSubplotsState,
) -> FigureOperationState:
    squeezed = data.squeeze(drop=True)
    if squeezed.ndim == 1:
        return FigureOperationState.line(
            label=_source_label(data),
            source=name,
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        )

    slice_dim: str | None = None
    slice_values: tuple[float, ...] = ()
    if squeezed.ndim > 2:
        slice_dim = str(squeezed.dims[0])
        value = _middle_coord_value(squeezed, slice_dim)
        if value is not None:
            slice_values = (value,)

    return FigureOperationState.plot_slices(
        label=_source_label(data),
        sources=(name,),
        axes=FigureAxesSelectionState(axes=_all_axes(setup)),
        slice_dim=slice_dim,
        slice_values=slice_values,
    )


def _default_setup_for_data(data: xr.DataArray) -> FigureSubplotsState:
    squeezed = data.squeeze(drop=True)
    if squeezed.ndim <= 2:
        return FigureSubplotsState()
    return FigureSubplotsState(nrows=1, ncols=1)


def _decode_indexer(value: typing.Any) -> typing.Any:
    if isinstance(value, dict) and value.get("kind") == "slice":
        return slice(value.get("start"), value.get("stop"), value.get("step"))
    return value


def _decode_indexers(indexers: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    return {key: _decode_indexer(value) for key, value in indexers.items()}
