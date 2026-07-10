"""Source and data-selection helpers for Figure Composer."""

from __future__ import annotations

import contextlib
import re
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

import erlab.interactive.imagetool.slicer
import erlab.interactive.utils
from erlab.interactive._figurecomposer._axes import _all_axes
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
    FigureOperationState,
    FigureSourceState,
    FigureSubplotsState,
    _restore_slice_mapping,
)

# Keep every binding emitted by Figure Composer code generation here. Source aliases
# share the generated module namespace, so missing one can silently replace source data.
_FIGURE_CODE_IMPORT_BINDINGS = frozenset(
    {
        "_erlab_stylesheets",
        "eplt",
        "erlab",
        "mcolors",
        "mtransforms",
        "np",
        "plt",
        "sns",
        "xr",
        "xarray",
    }
)
_FIGURE_CODE_SETUP_BINDINGS = frozenset({"ax", "axs", "fig", "gs0"})
_FIGURE_CODE_OPERATION_BINDINGS = frozenset(
    {
        "_",
        "_line",
        "avec",
        "bvec",
        "color",
        "i",
        "index",
        "kz",
        "kz_values",
        "label",
        "line_color_values",
        "line_color_values_norm",
        "line_color_values_vmax",
        "line_color_values_vmin",
        "line_colors",
        "map_index",
        "offset",
        "plot_maps",
        "profile",
        "profile_data",
        "profiles",
        "scale",
        "slice_index",
        "slice_value",
        "source",
        "target_axes",
    }
)
_FIGURE_CODE_RESERVED_NAMES = (
    _FIGURE_CODE_IMPORT_BINDINGS
    | _FIGURE_CODE_SETUP_BINDINGS
    | _FIGURE_CODE_OPERATION_BINDINGS
)
_FIGURE_CODE_RESERVED_NAME_PATTERN = re.compile(
    r"(?:ax\d+|gs\d+(?:_\d+)*|_line(?:_\d+)?)"
)
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _source_alias_candidate(data: xr.DataArray) -> str | None:
    if data.name is None:
        return None
    text = str(data.name).strip()
    if not text:
        return None
    if erlab.interactive.utils._is_kwarg_name(text):
        return text
    if not any(character.isalnum() for character in text):
        return None
    snake_text = _CAMEL_CASE_BOUNDARY.sub("_", text).lower()
    alias = erlab.interactive.utils.IdentifierValidator().fixup(snake_text)
    if not erlab.interactive.utils._is_kwarg_name(alias):
        return None
    return alias


def _source_name(data: xr.DataArray) -> str:
    return _source_unique_name(_source_alias_candidate(data) or "data", set())


def _source_label(data: xr.DataArray) -> str:
    if data.name is None:
        return "data"
    return str(data.name)


def _source_display_label(
    source: FigureSourceState | None, name: str, *, disambiguate: bool = False
) -> str:
    """Return the user-facing source alias."""
    del source, disambiguate
    return name


def _source_display_tooltip(source: FigureSourceState | None, name: str) -> str:
    del source
    return f"Alias: {name}"


def _valid_source_variable(name: str) -> str:
    if not erlab.interactive.utils._is_kwarg_name(name):
        raise ValueError(f"Figure source name {name!r} is not a valid variable name")
    return name


def _source_alias_error(alias: str) -> str | None:
    if not alias:
        return "Source alias must not be empty."
    try:
        _valid_source_variable(alias)
    except ValueError:
        return f"{alias!r} is not a valid Python variable name."
    if (
        alias in _FIGURE_CODE_RESERVED_NAMES
        or _FIGURE_CODE_RESERVED_NAME_PATTERN.fullmatch(alias) is not None
    ):
        return (
            f"{alias!r} is used internally by generated Figure Composer code. "
            "Choose a different source alias."
        )
    return None


def _source_unique_name(source_name: str, reserved: set[str]) -> str:
    base = (
        f"{source_name}_source"
        if _FIGURE_CODE_RESERVED_NAME_PATTERN.fullmatch(source_name) is not None
        else source_name
    )
    alias = base
    suffix = 2
    while _source_alias_error(alias) is not None or alias in reserved:
        alias = f"{base}_{suffix}"
        suffix += 1
    reserved.add(alias)
    return alias


def _public_source_data(data: xr.DataArray) -> xr.DataArray:
    return erlab.interactive.imagetool.slicer.restore_nonuniform_dims(data)


def _available_source_dims(
    source_data: Mapping[str, xr.DataArray], sources: Sequence[str]
) -> list[str]:
    dims: list[str] = []
    for source in sources:
        data = source_data.get(source)
        if data is None:
            continue
        for dim in _public_source_data(data).dims:
            if dim not in dims:
                dims.append(str(dim))
    return dims


def _selected_data(
    source_data: Mapping[str, xr.DataArray], selection: FigureDataSelectionState
) -> xr.DataArray | None:
    selected = source_data.get(selection.source)
    if selected is None:
        return None
    selected = _public_source_data(selected)
    if selection.isel:
        selected = selected.isel(_decode_indexers(selection.isel))
    if selection.qsel:
        selected = selected.qsel(_decode_indexers(selection.qsel))
    if selection.mean_dims:
        mean_arg: str | tuple[str, ...]
        if len(selection.mean_dims) == 1:
            mean_arg = selection.mean_dims[0]
        else:
            mean_arg = selection.mean_dims
        selected = selected.qsel.mean(mean_arg)
    return selected


def _source_selection(source: FigureSourceState) -> FigureDataSelectionState:
    return FigureDataSelectionState(
        source=source.name,
        isel=dict(source.isel),
        qsel=dict(source.qsel),
        mean_dims=tuple(source.mean_dims),
    )


def _source_has_selection(source: FigureSourceState) -> bool:
    return bool(source.isel or source.qsel or source.mean_dims)


def _source_with_selection(
    source: FigureSourceState, selection: FigureDataSelectionState
) -> FigureSourceState:
    has_selection = bool(selection.isel or selection.qsel or selection.mean_dims)
    return source.model_copy(
        update={
            "isel": dict(selection.isel),
            "qsel": dict(selection.qsel),
            "mean_dims": tuple(selection.mean_dims),
            "selection_source": (
                (source.selection_source or source.name) if has_selection else None
            ),
        }
    )


def _selected_source_data(
    data: xr.DataArray, source: FigureSourceState
) -> xr.DataArray:
    selected = _selected_data({source.name: data}, _source_selection(source))
    if selected is None:  # pragma: no cover
        raise KeyError(source.name)
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
    data = _public_source_data(data)
    squeezed = data.squeeze(drop=True)
    if squeezed.ndim == 1:
        return FigureOperationState.line(
            label=_source_label(data),
            source=name,
            axes=FigureAxesSelectionState(axes=((0, 0),)),
        )
    if squeezed.ndim == 2:
        return FigureOperationState.plot_array(
            label=_source_label(data),
            source=name,
            axes=FigureAxesSelectionState(axes=_all_axes(setup)[:1]),
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
    data = _public_source_data(data)
    squeezed = data.squeeze(drop=True)
    if squeezed.ndim <= 2:
        return FigureSubplotsState()
    return FigureSubplotsState(nrows=1, ncols=1)


def _decode_indexers(indexers: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    return _restore_slice_mapping(indexers, allow_legacy=True)
