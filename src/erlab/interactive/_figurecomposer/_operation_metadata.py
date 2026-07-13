"""Qt-widget-independent metadata derived from Figure Composer operations."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._custom_code import _custom_code_names
from erlab.interactive._figurecomposer._state import (
    FigureMethodFamily,
    FigureOperationKind,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive._figurecomposer._state import FigureOperationState


def _unique_source_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def is_axes_plot_data_method(family: FigureMethodFamily, name: str) -> bool:
    """Return whether a method can plot values selected from recipe sources."""
    return family == FigureMethodFamily.AXES and name in {"plot", "errorbar"}


def is_axes_errorbar_data_method(family: FigureMethodFamily, name: str) -> bool:
    """Return whether a source-backed method accepts error values."""
    return family == FigureMethodFamily.AXES and name == "errorbar"


def declared_operation_source_names(
    operation: FigureOperationState,
) -> tuple[str, ...]:
    """Return data sources explicitly selected by an operation's controls."""
    if operation.kind in {
        FigureOperationKind.PLOT_ARRAY,
        FigureOperationKind.PLOT_SLICES,
    }:
        return _unique_source_names(operation.sources)
    if operation.kind == FigureOperationKind.LINE:
        if len(operation.map_selections) > 1:
            return _unique_source_names(
                selection.source for selection in operation.map_selections
            )
        return (operation.line_source,) if operation.line_source is not None else ()
    if operation.kind == FigureOperationKind.PHOTON_ENERGY_OVERLAY:
        return (
            (operation.hv_overlay_source,)
            if operation.hv_overlay_source is not None
            else ()
        )
    if (
        operation.kind == FigureOperationKind.METHOD
        and is_axes_plot_data_method(operation.method_family, operation.method_name)
        and operation.method_plot_data_mode == "from_data"
    ):
        values = [operation.method_plot_x, operation.method_plot_y]
        if is_axes_errorbar_data_method(operation.method_family, operation.method_name):
            values.extend((operation.method_plot_xerr, operation.method_plot_yerr))
        return _unique_source_names(
            value.source for value in values if value is not None
        )
    return ()


def recipe_operation_source_names(
    operation: FigureOperationState,
    available_source_names: Iterable[str],
) -> tuple[str, ...]:
    """Return every recipe source read by an operation during execution."""
    names = list(declared_operation_source_names(operation))
    if operation.kind == FigureOperationKind.CUSTOM:
        loaded_names = _custom_code_names(operation.code)
        names.extend(name for name in available_source_names if name in loaded_names)
    return _unique_source_names(names)
