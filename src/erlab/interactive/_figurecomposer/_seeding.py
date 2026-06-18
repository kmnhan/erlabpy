"""Helpers for seeding Figure Composer recipes from existing interactive state."""

from __future__ import annotations

import typing

import numpy as np

import erlab
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
)

if typing.TYPE_CHECKING:
    import xarray as xr


def _plot_slices_style_update(operation: FigureOperationState) -> dict[str, typing.Any]:
    norm_gamma = operation.norm_gamma
    if norm_gamma is None:
        norm_gamma = operation.gamma
    return {
        "cmap": operation.cmap,
        "norm_name": operation.norm_name or "PowerNorm",
        "norm_gamma": norm_gamma,
        "norm_clip": operation.norm_clip,
        "norm_kwargs": dict(operation.norm_kwargs),
        "vmin": operation.vmin,
        "vmax": operation.vmax,
        "vcenter": operation.vcenter,
        "halfrange": operation.halfrange,
    }


def _plot_slices_neutral_norm_update() -> dict[str, typing.Any]:
    return {
        "same_limits": False,
        "gamma": None,
        "norm_name": "PowerNorm",
        "norm_gamma": None,
        "norm_clip": None,
        "norm_kwargs": {},
        "vmin": None,
        "vmax": None,
        "vcenter": None,
        "halfrange": None,
    }


def _plot_slices_panel_norm_style_update(
    style: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    updates: dict[str, typing.Any] = {}
    norm_name = style["norm_name"]
    if norm_name != "PowerNorm":
        updates["norm_name"] = norm_name
    for key in (
        "norm_gamma",
        "norm_clip",
        "vmin",
        "vmax",
        "vcenter",
        "halfrange",
    ):
        if style[key] is not None:
            updates[key] = style[key]
    if style["norm_kwargs"]:
        updates["norm_kwargs"] = dict(style["norm_kwargs"])
    return updates


def plot_slices_operation_with_source_styles(
    operation: FigureOperationState,
    source_operations: tuple[FigureOperationState, ...],
    *,
    selections_per_source: int,
) -> FigureOperationState:
    """Return ``operation`` with panel styles seeded from source operations."""
    if len(source_operations) <= 1:
        return operation

    styles = tuple(
        _plot_slices_style_update(source_operation)
        for source_operation in source_operations
    )
    first_style = styles[0]
    cmap_differs = any(style["cmap"] != first_style["cmap"] for style in styles[1:])
    norm_fields = (
        "norm_name",
        "norm_gamma",
        "norm_clip",
        "norm_kwargs",
        "vmin",
        "vmax",
        "vcenter",
        "halfrange",
    )
    norm_differs = any(
        any(style[field] != first_style[field] for field in norm_fields)
        for style in styles[1:]
    )
    if not cmap_differs and not norm_differs:
        return operation

    updates: dict[str, typing.Any] = {"panel_styles_enabled": True}
    if cmap_differs:
        updates["cmap"] = None
    if norm_differs:
        updates.update(_plot_slices_neutral_norm_update())

    panel_styles: list[FigurePlotSlicesPanelStyleState] = []
    slice_count = max(len(operation.slice_values), 1)
    for source_index, style in enumerate(styles):
        for selection_index in range(selections_per_source):
            map_index = source_index * selections_per_source + selection_index
            for slice_index in range(slice_count):
                style_update: dict[str, typing.Any] = {}
                if cmap_differs and style["cmap"] is not None:
                    style_update["cmap"] = style["cmap"]
                if norm_differs:
                    style_update.update(_plot_slices_panel_norm_style_update(style))
                if style_update:
                    panel_styles.append(
                        FigurePlotSlicesPanelStyleState(
                            map_index=map_index,
                            slice_index=slice_index,
                            **style_update,
                        )
                    )

    if not panel_styles:
        return operation

    return operation.model_copy(
        update={
            **updates,
            "panel_styles": tuple(panel_styles),
        }
    )


def _default_bz_updates() -> dict[str, typing.Any]:
    opts = erlab.interactive.options.model.ktool.bz
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
    from erlab.interactive._figurecomposer._sources import _public_source_data

    data = _public_source_data(data).squeeze(drop=True)
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
