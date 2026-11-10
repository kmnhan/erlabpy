"""Helpers for seeding Figure Composer recipes from existing interactive state."""

from __future__ import annotations

import typing

import numpy as np

from erlab.interactive._figurecomposer._defaults import _current_options
from erlab.interactive._figurecomposer._model._axes import _all_axes_for_shape
from erlab.interactive._figurecomposer._model._operation_metadata import (
    rename_operation_sources,
)
from erlab.interactive._figurecomposer._model._sources import _middle_coord_value
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureImageColorStyleState,
    FigureImageNormStyleState,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureSubplotsState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr


def image_operations_have_compatible_styles(
    operations: tuple[FigureOperationState, ...],
) -> bool:
    if not operations:
        return True
    first_style = operations[0].image_style.operation_style()
    return all(
        operation.image_style.operation_style() == first_style
        for operation in operations[1:]
    )


def plot_slices_operation_with_source_styles(
    operation: FigureOperationState,
    source_operations: tuple[FigureOperationState, ...],
    *,
    selections_per_source: int,
) -> FigureOperationState:
    """Return ``operation`` with panel styles seeded from source operations."""
    if len(source_operations) <= 1:
        return operation

    if not image_operations_have_compatible_styles(source_operations):
        raise ValueError("Image operations have incompatible image styles")
    panels = []
    slice_count = max(len(operation.slice_values), 1)
    for source_index, source_operation in enumerate(source_operations):
        source_style = source_operation.image_style.color_style
        overrides = (
            {
                (style.map_index, style.slice_index): style
                for style in source_operation.panel_styles
            }
            if source_operation.panel_styles_enabled
            else {}
        )
        for selection_index in range(selections_per_source):
            map_index = source_index * selections_per_source + selection_index
            for slice_index in range(slice_count):
                style = source_style
                override = overrides.get((selection_index, slice_index))
                if override is not None:
                    style = style.model_copy(
                        update={
                            name: value
                            for name in FigureImageColorStyleState.model_fields
                            if (value := getattr(override, name)) is not None
                            and value != {}
                        }
                    )
                panels.append((map_index, slice_index, style))
    first_style = panels[0][2]
    cmap_differs = any(style.cmap != first_style.cmap for _, _, style in panels[1:])
    norm_differs = any(
        any(
            getattr(style, field) != getattr(first_style, field)
            for field in FigureImageNormStyleState.model_fields
        )
        for _, _, style in panels[1:]
    )
    if not cmap_differs and not norm_differs:
        if not any(
            source.panel_styles_enabled and source.panel_styles
            for source in source_operations
        ):
            return operation
        return operation.model_copy(
            update={
                **{
                    name: getattr(first_style, name)
                    for name in FigureImageColorStyleState.model_fields
                },
                "panel_styles_enabled": False,
                "panel_styles": (),
            }
        )

    updates: dict[str, typing.Any] = {"panel_styles_enabled": True}
    if cmap_differs:
        updates["cmap"] = None
    if norm_differs:
        updates.update(FigureImageNormStyleState(norm_name="PowerNorm").model_dump())
        updates.update(same_limits=False, gamma=None)

    panel_styles = []
    neutral_norm = FigureImageNormStyleState(norm_name="PowerNorm")
    for map_index, slice_index, style in panels:
        style_update: dict[str, typing.Any] = {}
        if cmap_differs and style.cmap is not None:
            style_update["cmap"] = style.cmap
        if norm_differs:
            style_update.update(
                {
                    name: value
                    for name in FigureImageNormStyleState.model_fields
                    if (value := getattr(style, name)) != getattr(neutral_norm, name)
                }
            )
        if style_update:
            panel_styles.append(
                FigurePlotSlicesPanelStyleState(
                    map_index=map_index,
                    slice_index=slice_index,
                    **style_update,
                )
            )
    return operation.model_copy(update={**updates, "panel_styles": tuple(panel_styles)})


def _default_bz_updates() -> dict[str, typing.Any]:
    opts = _current_options().ktool.bz
    return {
        "bz_a": opts.default_a,
        "bz_b": opts.default_b,
        "bz_c": opts.default_c,
        "bz_alpha": opts.default_alpha,
        "bz_beta": opts.default_beta,
        "bz_gamma": opts.default_gamma,
        "bz_centering_type": opts.default_centering,
        "bz_angle": opts.default_rot,
    }


def _coordinate_bounds(data: xr.DataArray, dim: str) -> tuple[float, float] | None:
    coord = data.coords.get(dim)
    if coord is None:
        return None
    try:
        values = np.asarray(coord.values, dtype=float)
    except (TypeError, ValueError):
        return None
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def _momentum_bounds(
    data: xr.DataArray, x_dim: str, y_dim: str
) -> tuple[float, float, float, float] | None:
    x_bounds = _coordinate_bounds(data, x_dim)
    y_bounds = _coordinate_bounds(data, y_dim)
    if x_bounds is None or y_bounds is None:
        return None
    return (*x_bounds, *y_bounds)


def _representative_coord_value(data: xr.DataArray, name: str) -> float | None:
    coord = data.coords.get(name)
    if coord is None:
        return None
    try:
        values = np.asarray(coord.values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.nanmedian(finite))


def bz_overlay_operation_from_momentum_data(
    data: xr.DataArray,
    *,
    axes: FigureAxesSelectionState | None = None,
) -> FigureOperationState | None:
    """Seed a BZ overlay from simple momentum-cut dimensions and bounds."""
    data = data.squeeze(drop=True)
    dims = {str(dim) for dim in data.dims}
    updates = _default_bz_updates()

    if {"kx", "ky"}.issubset(dims):
        bounds = _momentum_bounds(data, "kx", "ky")
        if bounds is None:
            return None
        return FigureOperationState.bz_overlay(axes=axes).model_copy(
            update={**updates, "bz_mode": "in_plane", "bz_bounds": bounds}
        )

    if "kz" not in dims:
        return None
    for k_parallel_dim in ("kx", "ky"):
        if k_parallel_dim not in dims:
            continue
        bounds = _momentum_bounds(data, k_parallel_dim, "kz")
        if bounds is None:
            return None
        return FigureOperationState.bz_overlay(
            axes=axes, mode="out_of_plane"
        ).model_copy(update={**updates, "bz_bounds": bounds})
    return None


def _flat_ktool_out_of_plane_value(
    data: xr.DataArray, operation: FigureOperationState
) -> float | None:
    if operation.bz_mode != "out_of_plane":
        return None
    dims = {str(dim) for dim in data.dims}
    k_parallel_dim = "kx" if "kx" in dims else "ky" if "ky" in dims else None
    if k_parallel_dim is None:
        return None
    fixed_dim = "ky" if k_parallel_dim == "kx" else "kx"
    return _representative_coord_value(data, fixed_dim)


def bz_overlay_operation_from_ktool(
    tool: typing.Any,
    data: xr.DataArray,
    *,
    axes: FigureAxesSelectionState | None = None,
) -> FigureOperationState | None:
    """Seed a BZ overlay from a ktool converted output and current ktool state."""
    status = tool.tool_status
    if not status.bz_enabled:
        return None

    operation = bz_overlay_operation_from_momentum_data(data, axes=axes)
    if operation is None:
        return None

    lattice_params = tuple(float(value) for value in status.lattice_params)
    updates: dict[str, typing.Any] = {
        "bz_a": lattice_params[0],
        "bz_b": lattice_params[1],
        "bz_c": lattice_params[2],
        "bz_alpha": lattice_params[3],
        "bz_beta": lattice_params[4],
        "bz_gamma": lattice_params[5],
        "bz_centering_type": status.centering,
        "bz_angle": float(status.rot),
        "bz_kz_pi_over_c": float(status.kz),
        "bz_vertices": bool(status.points),
        "bz_midpoints": bool(status.points),
        "line_kw": {"color": "m", "linestyle": "-", "linewidth": 2.0},
    }
    k_parallel = _flat_ktool_out_of_plane_value(data, operation)
    if k_parallel is not None:
        updates["bz_k_parallel"] = k_parallel
    return operation.model_copy(update=updates)


def _default_slice_selection(
    data: xr.DataArray,
) -> tuple[str | None, tuple[float, ...]]:
    slice_dim = None
    slice_values: tuple[float, ...] = ()
    if data.ndim > 2:
        slice_dim = str(data.dims[0])
        value = _middle_coord_value(data, slice_dim)
        if value is not None:
            slice_values = (value,)
    return slice_dim, slice_values


def _make_operations_for_sources(
    source_data: Mapping[str, xr.DataArray],
    *,
    setup: FigureSubplotsState,
) -> tuple[FigureOperationState, ...]:
    """Choose readable default operations for a group of selected sources."""
    if not source_data:
        return ()

    source_names = tuple(source_data)
    all_axes = FigureAxesSelectionState(
        axes=_all_axes_for_shape(setup.nrows, setup.ncols)
    )
    squeezed = [data.squeeze(drop=True) for data in source_data.values()]

    if all(data.ndim == 2 for data in squeezed):
        operations = []
        for index, source_name in enumerate(source_names):
            row = min(index, setup.nrows - 1)
            operations.append(
                FigureOperationState.plot_array(
                    label=source_name,
                    source=source_name,
                    axes=FigureAxesSelectionState(axes=((row, 0),)),
                )
            )
        return tuple(operations)

    if all(data.ndim > 1 for data in squeezed):
        slice_dim, slice_values = _default_slice_selection(squeezed[0])
        operation = FigureOperationState.plot_slices(
            label="plot_slices",
            sources=source_names,
            axes=all_axes,
            slice_dim=slice_dim,
            slice_values=slice_values,
        ).model_copy(update={"order": "F"} if len(source_names) > 1 else {})
        return (operation,)

    operations = []
    for index, (source_name, data) in enumerate(
        zip(source_names, squeezed, strict=True)
    ):
        row = min(index, setup.nrows - 1)
        if data.ndim == 1:
            operations.append(
                FigureOperationState.line(
                    label=source_name,
                    source=source_name,
                    axes=FigureAxesSelectionState(axes=((row, 0),)),
                )
            )
        elif data.ndim == 2:
            operations.append(
                FigureOperationState.plot_array(
                    label=source_name,
                    source=source_name,
                    axes=FigureAxesSelectionState(axes=((row, 0),)),
                )
            )
        else:
            slice_dim, slice_values = _default_slice_selection(data)
            operations.append(
                FigureOperationState.plot_slices(
                    label=source_name,
                    sources=(source_name,),
                    axes=FigureAxesSelectionState(axes=((row, 0),)),
                    slice_dim=slice_dim,
                    slice_values=slice_values,
                )
            )
    return tuple(operations)


def _plot_slices_grid_shape(operation: FigureOperationState) -> tuple[int, int]:
    map_count = (
        len(operation.map_selections)
        if operation.map_selections
        else len(operation.sources)
    )
    map_count = max(map_count, 1)
    slice_count = max(len(operation.slice_values), 1)
    if operation.order == "F":
        return slice_count, map_count
    return map_count, slice_count


def _setup_for_operation(
    operation: FigureOperationState | None,
    source_data: Mapping[str, xr.DataArray],
) -> FigureSubplotsState:
    if operation is not None and operation.kind == FigureOperationKind.PLOT_SLICES:
        nrows, ncols = _plot_slices_grid_shape(operation)
        return FigureSubplotsState(nrows=nrows, ncols=ncols)

    squeezed = [data.squeeze(drop=True) for data in source_data.values()]
    return FigureSubplotsState(nrows=max(len(squeezed), 1), ncols=1)


def _setup_for_image_operations(
    operations: tuple[FigureOperationState, ...],
    source_data: Mapping[str, xr.DataArray],
) -> FigureSubplotsState:
    if operations and any(
        operation.kind == FigureOperationKind.PLOT_SLICES for operation in operations
    ):
        shapes = tuple(
            _plot_slices_grid_shape(operation.model_copy(update={"order": "F"}))
            if operation.kind == FigureOperationKind.PLOT_SLICES
            else (1, 1)
            for operation in operations
        )
        return FigureSubplotsState(
            nrows=max(shape[0] for shape in shapes),
            ncols=sum(shape[1] for shape in shapes),
        )
    if operations and all(
        operation.kind == FigureOperationKind.PLOT_ARRAY
        and len(operation.map_selections) == 1
        for operation in operations
    ):
        return FigureSubplotsState(nrows=1, ncols=len(operations))
    return _setup_for_operation(None, source_data)


def _operation_with_source_names(
    operation: FigureOperationState, source_name_map: Mapping[str, str]
) -> FigureOperationState:
    if not source_name_map:
        return operation
    return rename_operation_sources(operation, source_name_map)


def _operations_with_append_axes(
    operations: tuple[FigureOperationState, ...],
    axes_selection: FigureAxesSelectionState,
) -> tuple[FigureOperationState, ...]:
    if (
        len(operations) > 1
        and not axes_selection.expression
        and all(
            operation.kind
            in {FigureOperationKind.PLOT_ARRAY, FigureOperationKind.PLOT_SLICES}
            for operation in operations
        )
    ):
        counts = tuple(
            max(len(operation.map_selections) or len(operation.sources), 1)
            * max(len(operation.slice_values), 1)
            if operation.kind == FigureOperationKind.PLOT_SLICES
            else 1
            for operation in operations
        )
        if axes_selection.axes and len(axes_selection.axes) >= sum(counts):
            if any(
                operation.kind == FigureOperationKind.PLOT_SLICES
                for operation in operations
            ):
                rows = sorted({row for row, _col in axes_selection.axes})
                columns = sorted({col for _row, col in axes_selection.axes})
                shapes = tuple(
                    _plot_slices_grid_shape(operation.model_copy(update={"order": "F"}))
                    if operation.kind == FigureOperationKind.PLOT_SLICES
                    else (1, 1)
                    for operation in operations
                )
                if len(rows) >= max(shape[0] for shape in shapes) and len(
                    columns
                ) >= sum(shape[1] for shape in shapes):
                    selected = []
                    offset = 0
                    for operation, (nrows, ncols) in zip(
                        operations, shapes, strict=True
                    ):
                        axes = tuple(
                            (row, col)
                            for row in rows[:nrows]
                            for col in columns[offset : offset + ncols]
                        )
                        if not set(axes).issubset(axes_selection.axes):
                            break
                        selected.append(
                            operation.model_copy(
                                update={
                                    "axes": FigureAxesSelectionState(axes=axes),
                                    **(
                                        {"order": "F"}
                                        if operation.kind
                                        == FigureOperationKind.PLOT_SLICES
                                        else {}
                                    ),
                                }
                            )
                        )
                        offset += ncols
                    else:
                        return tuple(selected)
            offset = 0
            selected = []
            for operation, count in zip(operations, counts, strict=True):
                axes = axes_selection.axes[offset : offset + count]
                selected.append(
                    operation.model_copy(
                        update={"axes": FigureAxesSelectionState(axes=axes)}
                    )
                )
                offset += count
            return tuple(selected)
        if axes_selection.axes_ids and len(axes_selection.axes_ids) >= sum(counts):
            offset = 0
            selected = []
            for operation, count in zip(operations, counts, strict=True):
                selected.append(
                    operation.model_copy(
                        update={
                            "axes": FigureAxesSelectionState(
                                axes_ids=axes_selection.axes_ids[
                                    offset : offset + count
                                ]
                            )
                        }
                    )
                )
                offset += count
            return tuple(selected)
    return tuple(
        operation.model_copy(update={"axes": axes_selection})
        for operation in operations
    )
