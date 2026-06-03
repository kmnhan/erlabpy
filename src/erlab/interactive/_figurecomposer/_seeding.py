"""Helpers for seeding Figure Composer recipes from existing interactive state."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._state import (
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
)


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
