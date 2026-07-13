"""Qt-widget-independent document model for Figure Composer."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_invalid_axes_ids,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._operation_metadata import (
    recipe_operation_source_names,
)
from erlab.interactive._figurecomposer._state import FigureOperationKind

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import xarray as xr

    from erlab.interactive._figurecomposer._state import (
        FigureAxesSelectionState,
        FigureOperationState,
        FigureRecipeState,
        FigureSourceState,
    )


class FigureRecipeContext(typing.Protocol):
    """Read-only recipe and source access used outside the Qt editor."""

    @property
    def recipe(self) -> FigureRecipeState: ...

    @property
    def source_data(self) -> Mapping[str, xr.DataArray]: ...

    def source_names(self) -> tuple[str, ...]: ...

    def source_by_name(self) -> dict[str, FigureSourceState]: ...

    def source_dependency_names(
        self,
        names: Iterable[str],
        *,
        stop_at: frozenset[str] = frozenset(),
        reject_cycles: bool = False,
    ) -> tuple[str, ...]: ...

    def direct_sources_used_by_recipe(
        self, *, enabled_only: bool = False, executable_only: bool = False
    ) -> set[str]: ...

    def axes_selection_has_invalid_target(
        self, selection: FigureAxesSelectionState
    ) -> bool: ...


class FigureDocument:
    """Mutable Figure Composer document without editor-widget dependencies."""

    def __init__(
        self,
        recipe: FigureRecipeState,
        *,
        source_data: Mapping[str, xr.DataArray] | None = None,
        source_selection_base_data: Mapping[str, xr.DataArray] | None = None,
    ) -> None:
        self.recipe = recipe
        self.source_data = dict(source_data or {})
        self.source_selection_base_data = dict(source_selection_base_data or {})

    def source_names(self) -> tuple[str, ...]:
        names = tuple(source.name for source in self.recipe.sources)
        return names or tuple(self.source_data)

    def source_by_name(self) -> dict[str, FigureSourceState]:
        return {source.name: source for source in self.recipe.sources}

    def operation_source_names(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return recipe_operation_source_names(operation, self.source_names())

    def source_dependency_names(
        self,
        names: Iterable[str],
        *,
        stop_at: frozenset[str] = frozenset(),
        reject_cycles: bool = False,
    ) -> tuple[str, ...]:
        """Return source names in dependency order, parents before children."""
        source_by_name = self.source_by_name()
        ordered: list[str] = []
        resolved: set[str] = set()
        resolving: list[str] = []

        def add_dependencies(name: str) -> None:
            if name in resolved:
                return
            if name in resolving:
                if reject_cycles:
                    cycle_start = resolving.index(name)
                    cycle = (*resolving[cycle_start:], name)
                    raise FigureComposerInputError(
                        "Cannot generate code because source selections contain a "
                        f"dependency cycle: {' -> '.join(cycle)}."
                    )
                return
            resolving.append(name)
            source = source_by_name.get(name)
            if source is not None and name not in stop_at:
                base_name = source.selection_source
                if base_name is not None and base_name != name:
                    add_dependencies(base_name)
            resolving.pop()
            resolved.add(name)
            ordered.append(name)

        for name in names:
            add_dependencies(name)
        return tuple(ordered)

    def operation_source_dependency_names(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return self.source_dependency_names(self.operation_source_names(operation))

    def direct_sources_used_by_recipe(
        self, *, enabled_only: bool = False, executable_only: bool = False
    ) -> set[str]:
        return {
            source_name
            for operation in self.recipe.operations
            if not enabled_only or operation.enabled
            if not executable_only
            or operation.kind != FigureOperationKind.CUSTOM
            or operation.trusted
            for source_name in self.operation_source_names(operation)
        }

    def axes_selection_has_invalid_target(
        self, selection: FigureAxesSelectionState
    ) -> bool:
        if selection.expression:
            return False
        setup = self.recipe.setup
        if setup.layout_mode == "gridspec":
            if not _gridspec_valid_axes_ids(setup, selection.axes_ids):
                return True
            return bool(_gridspec_invalid_axes_ids(setup, selection.axes_ids))
        return bool(selection.invalid_axes(setup)) or not bool(
            selection.valid_axes(setup)
        )
