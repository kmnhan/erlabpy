"""Qt-widget-independent document model for Figure Composer."""

from __future__ import annotations

import collections
import typing
import uuid

from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._model._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axes_root_targets,
    _gridspec_invalid_axes_ids,
    _gridspec_setup_from_subplots,
    _gridspec_valid_axes_ids,
    _subplots_setup_from_gridspec,
)
from erlab.interactive._figurecomposer._model._operation_metadata import (
    operation_uses_axes,
    recipe_operation_source_names,
    rename_operation_sources,
)
from erlab.interactive._figurecomposer._model._sources import (
    _selected_source_data,
    _source_alias_error,
    _source_has_selection,
    _source_selection,
    _source_unique_name,
    _source_with_selection,
    selection_has_effect,
    selection_with_dimension,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureSourceState,
    FigureSubplotsState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import (
        FigureAxesSelectionState,
        FigureOperationState,
        FigureRecipeState,
    )


class FigureSourceAddResult(typing.NamedTuple):
    """Outcome of adding or refreshing a batch of figure sources."""

    added: tuple[tuple[str, str], ...] = ()
    updated: tuple[tuple[str, str], ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.added or self.updated)

    @property
    def name_map(self) -> dict[str, str]:
        """Map accepted requested names to their stored recipe names."""
        return dict((*self.added, *self.updated))


class FigureSourceUpdateResult(typing.NamedTuple):
    """Outcome of updating existing source data."""

    updated: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.updated)


class FigureOperationPasteResult(typing.NamedTuple):
    """Outcome of inserting copied operations and their source dependencies."""

    operation_ids: tuple[str, ...]
    source_data_changed: bool


class FigureRecipeContext(typing.Protocol):
    """Read-only recipe and source access used outside the Qt editor."""

    @property
    def recipe(self) -> FigureRecipeState: ...

    @property
    def source_data(self) -> Mapping[str, xr.DataArray]: ...

    def source_names(self) -> tuple[str, ...]: ...

    def source_by_name(self) -> dict[str, FigureSourceState]: ...

    def operation_index(self, operation_id: str) -> int | None: ...

    def operation_by_id(self, operation_id: str) -> FigureOperationState | None: ...

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
        self._recipe_revision = 0
        self._source_revision = 0
        self._source_changed_callback: Callable[[int], None] | None = None
        self._recipe = recipe.model_copy(
            update={"sources": self.normalized_source_states(recipe.sources)}
        )
        self._validate_operation_id_sequence(self._recipe.operations)
        self._source_data: dict[str, xr.DataArray]
        self._source_selection_base_data: dict[str, xr.DataArray]
        self.replace_source_payloads(
            source_data or {}, source_selection_base_data or {}
        )

    @property
    def recipe(self) -> FigureRecipeState:
        """Current validated recipe for internal read-only use.

        Mutations must go through the document replacement methods so revisions,
        history, rendering, and persistence remain synchronized.
        """
        return self._recipe

    @property
    def recipe_revision(self) -> int:
        """Monotonic revision for changes to the figure recipe."""
        return self._recipe_revision

    @property
    def source_data(self) -> Mapping[str, xr.DataArray]:
        """Effective source payloads used by figure operations."""
        return self._source_data

    @property
    def source_selection_base_data(self) -> Mapping[str, xr.DataArray]:
        """Unselected parent payloads used to replay source selections."""
        return self._source_selection_base_data

    @property
    def source_revision(self) -> int:
        """Monotonic revision for atomic source-payload replacements."""
        return self._source_revision

    def set_source_changed_callback(
        self, callback: Callable[[int], None] | None
    ) -> None:
        """Set the observer notified after subsequent source-payload changes."""
        self._source_changed_callback = callback

    def replace_recipe(self, recipe: FigureRecipeState) -> bool:
        """Replace the complete recipe after validating document invariants."""
        if recipe == self._recipe:
            return False
        self._validate_operation_id_sequence(recipe.operations)
        self._recipe = recipe
        self._recipe_revision += 1
        return True

    def replace_source_payloads(
        self,
        source_data: Mapping[str, xr.DataArray],
        selection_base_data: Mapping[str, xr.DataArray],
    ) -> None:
        """Replace effective and selection-base source payloads together."""
        updated_source_data = dict(source_data)
        updated_selection_base_data = dict(selection_base_data)
        self._source_data = updated_source_data
        self._source_selection_base_data = updated_selection_base_data
        self.touch_source_payloads()

    def touch_source_payloads(self) -> None:
        """Record in-place changes to the current source payloads."""
        self._source_revision += 1
        if self._source_changed_callback is not None:
            self._source_changed_callback(self._source_revision)

    def replace_setup(self, setup: FigureSubplotsState) -> bool:
        """Replace the complete validated figure layout setup."""
        validated = FigureSubplotsState.model_validate(setup.model_dump(mode="python"))
        if validated == self.recipe.setup:
            return False
        self.replace_recipe(self.recipe.model_copy(update={"setup": validated}))
        return True

    def convert_layout_mode(self, mode: typing.Literal["subplots", "gridspec"]) -> bool:
        """Convert layout mode and retarget axes operations atomically."""
        setup = self.recipe.setup
        if mode == setup.layout_mode:
            return False
        if mode == "gridspec":
            updated_setup = _gridspec_setup_from_subplots(setup)
            axes_ids = _gridspec_all_axes_ids(updated_setup)
            axis_id_by_tuple = {
                (row, col): axes_ids[row * updated_setup.ncols + col]
                for row in range(updated_setup.nrows)
                for col in range(updated_setup.ncols)
                if row * updated_setup.ncols + col < len(axes_ids)
            }

            def retarget(operation: FigureOperationState) -> FigureOperationState:
                if not operation_uses_axes(operation):
                    return operation
                axes = operation.axes.model_copy(
                    update={
                        "axes_ids": tuple(
                            axis_id_by_tuple[axis]
                            for axis in operation.axes.axes
                            if axis in axis_id_by_tuple
                        ),
                        "expression": "",
                    }
                )
                return operation.model_copy(update={"axes": axes})

        elif mode == "subplots":
            updated_setup = _subplots_setup_from_gridspec(setup)
            root_targets = _gridspec_axes_root_targets(setup)

            def retarget(operation: FigureOperationState) -> FigureOperationState:
                if not operation_uses_axes(operation):
                    return operation
                axes_targets = tuple(
                    dict.fromkeys(
                        target
                        for axes_id in operation.axes.axes_ids
                        for target in root_targets.get(axes_id, ())
                    )
                )
                axes = operation.axes.model_copy(
                    update={
                        "axes": axes_targets or operation.axes.axes or ((0, 0),),
                        "expression": "",
                    }
                )
                return operation.model_copy(update={"axes": axes})

        else:
            raise ValueError(f"unknown figure layout mode {mode!r}")
        operations = tuple(retarget(operation) for operation in self.recipe.operations)
        return self.replace_recipe(
            self.recipe.model_copy(
                update={"setup": updated_setup, "operations": operations}
            )
        )

    @staticmethod
    def _validate_operation_id_sequence(
        operations: Sequence[FigureOperationState],
    ) -> None:
        operation_ids = tuple(operation.operation_id for operation in operations)
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("operation IDs must be unique")

    def source_names(self) -> tuple[str, ...]:
        names = tuple(source.name for source in self.recipe.sources)
        return names or tuple(self.source_data)

    def source_by_name(self) -> dict[str, FigureSourceState]:
        return {source.name: source for source in self.recipe.sources}

    def source_alias_error(
        self, alias: str, *, current: str | None = None
    ) -> str | None:
        """Return why *alias* cannot identify a source in this document."""
        if error := _source_alias_error(alias):
            return error
        if alias != current and (
            alias in {source.name for source in self.recipe.sources}
            or alias in self.source_data
        ):
            return f"Source alias {alias!r} is already in use."
        return None

    @staticmethod
    def source_with_name(source: FigureSourceState, name: str) -> FigureSourceState:
        """Return *source* with its recipe-facing name changed."""
        selection_source = source.selection_source
        if selection_source == source.name:
            selection_source = None
        if source.name == name and selection_source == source.selection_source:
            return source
        updates: dict[str, typing.Any] = {
            "name": name,
            "selection_source": selection_source,
        }
        if source.label == source.name:
            updates["label"] = name
        return source.model_copy(update=updates)

    @classmethod
    def source_with_renamed_references(
        cls,
        source: FigureSourceState,
        rename_map: Mapping[str, str],
    ) -> FigureSourceState:
        """Rename one source and every explicit parent reference atomically."""
        old_name = source.name
        renamed = cls.source_with_name(source, rename_map.get(old_name, old_name))
        parent_name = source.selection_source
        if parent_name == old_name:
            parent_name = None
        elif parent_name is not None:
            parent_name = rename_map.get(parent_name, parent_name)
        if renamed.selection_source == parent_name:
            return renamed
        return renamed.model_copy(update={"selection_source": parent_name})

    @staticmethod
    def source_lineage_names(
        source_name: str,
        source_by_name: Mapping[str, FigureSourceState],
    ) -> tuple[str, ...]:
        """Return one source lineage from its available root to its leaf."""
        lineage: list[str] = []
        seen: set[str] = set()
        current_name = source_name
        while True:
            if current_name in seen:
                raise ValueError(
                    f"source selection dependency cycle includes {current_name!r}"
                )
            seen.add(current_name)
            lineage.append(current_name)
            current = source_by_name.get(current_name)
            if current is None or current.selection_source is None:
                break
            parent_name = current.selection_source
            if parent_name not in source_by_name:
                break
            current_name = parent_name
        lineage.reverse()
        return tuple(lineage)

    @staticmethod
    def source_states_with_propagated_link_metadata(
        source_by_name: Mapping[str, FigureSourceState],
        changed_names: Iterable[str],
    ) -> dict[str, FigureSourceState]:
        """Propagate origin-link metadata through selected descendants."""
        sources = dict(source_by_name)
        queue: collections.deque[str] = collections.deque(changed_names)
        visited = set(queue)
        while queue:
            parent_name = queue.popleft()
            parent = sources.get(parent_name)
            if parent is None:
                continue
            for source_name, source in tuple(sources.items()):
                if (
                    source_name in visited
                    or source.selection_source != parent_name
                    or source_name == parent_name
                ):
                    continue
                updates = {
                    field: getattr(parent, field)
                    for field in (
                        "node_uid",
                        "node_snapshot_token",
                        "data_role",
                        "provenance_spec",
                    )
                    if getattr(source, field) != getattr(parent, field)
                }
                if updates:
                    source = source.model_copy(update=updates)
                    sources[source_name] = source
                visited.add(source_name)
                queue.append(source_name)
        return sources

    @classmethod
    def normalized_source_states(
        cls, sources: Sequence[FigureSourceState]
    ) -> tuple[FigureSourceState, ...]:
        """Return source states satisfying selection graph invariants."""
        canonical = tuple(
            cls.source_with_name(source, source.name) for source in sources
        )
        source_by_name = {source.name: source for source in canonical}
        roots = tuple(
            source.name
            for source in canonical
            if source.selection_source is None
            or source.selection_source not in source_by_name
        )
        propagated = cls.source_states_with_propagated_link_metadata(
            source_by_name, roots
        )
        return tuple(propagated[source.name] for source in canonical)

    def operation_index(self, operation_id: str) -> int | None:
        """Return the recipe index for *operation_id*, if it is present."""
        for index, operation in enumerate(self.recipe.operations):
            if operation.operation_id == operation_id:
                return index
        return None

    def operation_by_id(self, operation_id: str) -> FigureOperationState | None:
        """Return the operation identified by *operation_id*, if it is present."""
        index = self.operation_index(operation_id)
        if index is None:
            return None
        return self.recipe.operations[index]

    def _operation_indices(self, indices: Iterable[int]) -> tuple[int, ...]:
        """Validate and normalize operation indices into recipe order."""
        requested = set(indices)
        operation_count = len(self.recipe.operations)
        if any(index < 0 or index >= operation_count for index in requested):
            raise IndexError("operation index out of range")
        return tuple(index for index in range(operation_count) if index in requested)

    def _validate_new_operation_ids(
        self,
        operations: Sequence[FigureOperationState],
        *,
        replacing_index: int | None = None,
    ) -> None:
        self._validate_operation_id_sequence(operations)
        new_ids = tuple(operation.operation_id for operation in operations)
        existing_ids = {
            operation.operation_id
            for index, operation in enumerate(self.recipe.operations)
            if index != replacing_index
        }
        if existing_ids.intersection(new_ids):
            raise ValueError("operation ID is already in use")

    def update_operations_by_ids(
        self,
        operation_ids: Iterable[str],
        updater: Callable[[int, FigureOperationState], FigureOperationState],
    ) -> bool:
        """Update matching operations in recipe order."""
        operation_id_set = set(operation_ids)
        if not operation_id_set:
            return False
        operations = list(self.recipe.operations)
        changed = False
        for index, operation in enumerate(operations):
            if operation.operation_id not in operation_id_set:
                continue
            updated = updater(index, operation)
            if updated.operation_id != operation.operation_id:
                raise ValueError("an in-place operation update cannot change its ID")
            if updated == operation:
                continue
            operations[index] = updated
            changed = True
        if not changed:
            return False
        self.replace_recipe(
            self.recipe.model_copy(update={"operations": tuple(operations)})
        )
        return True

    def replace_operation(self, index: int, operation: FigureOperationState) -> bool:
        """Replace one operation at *index*."""
        (index,) = self._operation_indices((index,))
        operations = list(self.recipe.operations)
        if operations[index] == operation:
            return False
        self._validate_new_operation_ids((operation,), replacing_index=index)
        operations[index] = operation
        self.replace_recipe(
            self.recipe.model_copy(update={"operations": tuple(operations)})
        )
        return True

    def replace_operations(self, operations: Sequence[FigureOperationState]) -> bool:
        """Replace the complete operation sequence after validating identities."""
        updated = tuple(operations)
        self._validate_operation_id_sequence(updated)
        if updated == self.recipe.operations:
            return False
        self.replace_recipe(self.recipe.model_copy(update={"operations": updated}))
        return True

    def append_operation(self, operation: FigureOperationState) -> int:
        """Append *operation* and return its recipe index."""
        index = len(self.recipe.operations)
        self.insert_operations(index, (operation,))
        return index

    def insert_operations(
        self, index: int, operations: Sequence[FigureOperationState]
    ) -> tuple[str, ...]:
        """Insert caller-created operations at *index* and return their IDs."""
        if index < 0 or index > len(self.recipe.operations):
            raise IndexError("operation insertion index out of range")
        inserted = tuple(operations)
        if not inserted:
            return ()
        self._validate_new_operation_ids(inserted)
        recipe_operations = list(self.recipe.operations)
        recipe_operations[index:index] = inserted
        self.replace_recipe(
            self.recipe.model_copy(update={"operations": tuple(recipe_operations)})
        )
        return tuple(operation.operation_id for operation in inserted)

    def _operation_copies(
        self, operations: Sequence[FigureOperationState]
    ) -> tuple[FigureOperationState, ...]:
        reserved_ids = {operation.operation_id for operation in self.recipe.operations}
        copies: list[FigureOperationState] = []
        for operation in operations:
            operation_id = uuid.uuid4().hex
            while operation_id in reserved_ids:  # pragma: no cover - UUID collision
                operation_id = uuid.uuid4().hex
            reserved_ids.add(operation_id)
            copies.append(
                operation.model_copy(update={"operation_id": operation_id}, deep=True)
            )
        return tuple(copies)

    def remove_operation_indices(self, indices: Sequence[int]) -> tuple[str, ...]:
        """Remove recipe indices and return removed IDs in recipe order."""
        index_set = set(self._operation_indices(indices))
        if not index_set:
            return ()
        removed = tuple(
            operation.operation_id
            for index, operation in enumerate(self.recipe.operations)
            if index in index_set
        )
        operations = tuple(
            operation
            for index, operation in enumerate(self.recipe.operations)
            if index not in index_set
        )
        self.replace_recipe(self.recipe.model_copy(update={"operations": operations}))
        return removed

    def duplicate_operations(self, indices: Sequence[int]) -> tuple[str, ...]:
        """Deep-copy selected operations with fresh identities."""
        ordered_indices = self._operation_indices(indices)
        if not ordered_indices:
            return ()
        duplicate_operations = self._operation_copies(
            tuple(self.recipe.operations[index] for index in ordered_indices)
        )
        return self.insert_operations(ordered_indices[-1] + 1, duplicate_operations)

    def paste_operations(
        self,
        index: int,
        operations: Sequence[FigureOperationState],
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
        selection_base_data: Mapping[str, xr.DataArray],
        *,
        preserve_existing: bool = False,
    ) -> FigureOperationPasteResult:
        """Insert copied operations and source dependencies as one transaction."""
        if index < 0 or index > len(self.recipe.operations):
            raise IndexError("operation insertion index out of range")

        existing_source_names = {source.name for source in self.recipe.sources}
        reserved_source_names = set(existing_source_names)
        reserved_source_names.update(self.source_data)
        unique_sources: list[FigureSourceState] = []
        seen_sources: set[str] = set()
        for source in sources:
            if source.name in seen_sources:
                continue
            seen_sources.add(source.name)
            unique_sources.append(source)

        rename_map: dict[str, str] = {}
        for source in unique_sources:
            if preserve_existing and source.name in reserved_source_names:
                rename_map[source.name] = source.name
                continue
            pasted_name = source.name
            if pasted_name in reserved_source_names:
                stem = f"{source.name}_copy"
                pasted_name = stem
                suffix = 2
                while pasted_name in reserved_source_names:
                    pasted_name = f"{stem}_{suffix}"
                    suffix += 1
            reserved_source_names.add(pasted_name)
            rename_map[source.name] = pasted_name

        source_list = list(self.recipe.sources)
        renamed_source_data: dict[str, xr.DataArray] = {}
        for source in unique_sources:
            pasted_name = rename_map[source.name]
            renamed_source = self.source_with_renamed_references(source, rename_map)
            if pasted_name not in existing_source_names:
                source_list.append(renamed_source)
                existing_source_names.add(pasted_name)
            if source.name in source_data and (
                pasted_name not in self.source_data or not preserve_existing
            ):
                renamed_source_data[pasted_name] = source_data[source.name]

        copied_operations = self._operation_copies(
            tuple(
                rename_operation_sources(operation, rename_map)
                for operation in operations
            )
        )
        self._validate_new_operation_ids(copied_operations)
        operation_list = list(self.recipe.operations)
        operation_list[index:index] = copied_operations

        renamed_selection_base_data: dict[str, xr.DataArray] = {}
        for source_name, data in selection_base_data.items():
            pasted_name = rename_map.get(source_name, source_name)
            if pasted_name not in existing_source_names:
                continue
            if preserve_existing and pasted_name in self.source_selection_base_data:
                continue
            renamed_selection_base_data[pasted_name] = data
        updated_source_data = dict(self.source_data)
        updated_source_data.update(renamed_source_data)
        updated_selection_base_data = dict(self.source_selection_base_data)
        updated_selection_base_data.update(renamed_selection_base_data)
        updated_recipe = self.recipe.model_copy(
            update={
                "sources": tuple(source_list),
                "operations": tuple(operation_list),
            }
        )

        self.replace_recipe(updated_recipe)
        self.replace_source_payloads(updated_source_data, updated_selection_base_data)
        return FigureOperationPasteResult(
            tuple(operation.operation_id for operation in copied_operations),
            bool(renamed_source_data or renamed_selection_base_data),
        )

    def reorder_operations(self, ordered_ids: Sequence[str]) -> bool:
        """Reorder operations using an exact permutation of their IDs."""
        operation_by_id = {
            operation.operation_id: operation for operation in self.recipe.operations
        }
        current_ids = tuple(
            operation.operation_id for operation in self.recipe.operations
        )
        ids = tuple(ordered_ids)
        if (
            len(operation_by_id) != len(current_ids)
            or len(ids) != len(current_ids)
            or len(set(ids)) != len(ids)
            or set(ids) != set(current_ids)
        ):
            raise ValueError("operation order must be an exact permutation")
        if ids == current_ids:
            return False
        self.replace_recipe(
            self.recipe.model_copy(
                update={
                    "operations": tuple(
                        operation_by_id[operation_id] for operation_id in ids
                    )
                }
            )
        )
        return True

    def can_move_operations(self, operation_ids: Iterable[str], offset: int) -> bool:
        """Return whether any selected operation can move by one position."""
        if offset not in {-1, 1}:
            raise ValueError("operation move offset must be -1 or 1")
        selected_ids = set(operation_ids)
        indices = tuple(
            index
            for index, operation in enumerate(self.recipe.operations)
            if operation.operation_id in selected_ids
        )
        if not indices:
            return False
        index_set = set(indices)
        if offset < 0:
            return any(index > 0 and index - 1 not in index_set for index in indices)
        return any(
            index < len(self.recipe.operations) - 1 and index + 1 not in index_set
            for index in indices
        )

    def move_operations(self, operation_ids: Iterable[str], offset: int) -> bool:
        """Move selected operations one position, preserving relative order."""
        if offset not in {-1, 1}:
            raise ValueError("operation move offset must be -1 or 1")
        selected_ids = set(operation_ids)
        indices = tuple(
            index
            for index, operation in enumerate(self.recipe.operations)
            if operation.operation_id in selected_ids
        )
        if not indices:
            return False
        operations = list(self.recipe.operations)
        index_set = set(indices)
        moved = False
        if offset < 0:
            for index in indices:
                if index > 0 and index - 1 not in index_set:
                    operations[index - 1], operations[index] = (
                        operations[index],
                        operations[index - 1],
                    )
                    index_set.remove(index)
                    index_set.add(index - 1)
                    moved = True
        else:
            for index in reversed(indices):
                if index < len(operations) - 1 and index + 1 not in index_set:
                    operations[index + 1], operations[index] = (
                        operations[index],
                        operations[index + 1],
                    )
                    index_set.remove(index)
                    index_set.add(index + 1)
                    moved = True
        if not moved:
            return False
        self.replace_recipe(
            self.recipe.model_copy(update={"operations": tuple(operations)})
        )
        return True

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
        return self._source_dependency_names(
            names,
            source_by_name=self.source_by_name(),
            stop_at=stop_at,
            reject_cycles=reject_cycles,
        )

    @staticmethod
    def _source_dependency_names(
        names: Iterable[str],
        *,
        source_by_name: Mapping[str, FigureSourceState],
        stop_at: frozenset[str] = frozenset(),
        reject_cycles: bool = False,
    ) -> tuple[str, ...]:
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

    def _source_used_by_operation(
        self, name: str, sources: Sequence[FigureSourceState]
    ) -> bool:
        source_by_name = {source.name: source for source in sources}
        return any(
            name
            in self._source_dependency_names(
                self.operation_source_names(operation),
                source_by_name=source_by_name,
            )
            for operation in self.recipe.operations
        )

    def source_usage_count(self, name: str) -> int:
        """Return the number of operations that depend on *name*."""
        return sum(
            name in self.operation_source_dependency_names(operation)
            for operation in self.recipe.operations
        )

    def sources_used_by_recipe(self) -> set[str]:
        """Return every direct and transitive source dependency in the recipe."""
        return {
            source_name
            for operation in self.recipe.operations
            for source_name in self.operation_source_dependency_names(operation)
        }

    def source_is_removable(self, name: str) -> bool:
        """Return whether *name* can be removed without dangling references."""
        return self._source_is_removable(name, self.recipe.sources)

    def _source_is_removable(
        self, name: str, sources: Sequence[FigureSourceState]
    ) -> bool:
        source_names = {source.name for source in sources}
        return (
            name in source_names
            and len(sources) > 1
            and not self._source_used_by_operation(name, sources)
            and not any(
                source.name != name and source.selection_source == name
                for source in sources
            )
        )

    def rename_source(self, old_name: str, new_name: str) -> bool:
        """Rename a source and every recipe reference to it atomically."""
        if old_name == new_name:
            return False
        if error := self.source_alias_error(new_name, current=old_name):
            raise FigureComposerInputError(error)
        if old_name not in {source.name for source in self.recipe.sources}:
            return False

        rename_map = {old_name: new_name}
        sources = tuple(
            self.source_with_renamed_references(source, rename_map)
            for source in self.recipe.sources
        )
        operations = tuple(
            rename_operation_sources(operation, rename_map)
            for operation in self.recipe.operations
        )
        source_data = {
            rename_map.get(name, name): data for name, data in self.source_data.items()
        }
        selection_base_data = {
            rename_map.get(name, name): data
            for name, data in self.source_selection_base_data.items()
        }
        updates: dict[str, typing.Any] = {
            "sources": sources,
            "operations": operations,
        }
        if self.recipe.primary_source == old_name:
            updates["primary_source"] = new_name

        self.replace_recipe(self.recipe.model_copy(update=updates))
        self.replace_source_payloads(source_data, selection_base_data)
        return True

    def duplicate_sources(self, names: Sequence[str]) -> tuple[str, ...]:
        """Duplicate sources in recipe order and return their new names."""
        selected_names = set(names)
        indices = tuple(
            index
            for index, source in enumerate(self.recipe.sources)
            if source.name in selected_names
        )
        if not indices:
            return ()
        sources = list(self.recipe.sources)
        source_data = dict(self.source_data)
        selection_base_data = dict(self.source_selection_base_data)
        reserved = {source.name for source in sources}
        reserved.update(source_data)
        duplicates: list[FigureSourceState] = []
        for index in indices:
            source = sources[index]
            alias = self._source_copy_alias(source.name, reserved)
            duplicates.append(self.source_with_name(source, alias))
            if source.name in source_data:
                source_data[alias] = source_data[source.name].copy(deep=False)
            if source.name in selection_base_data:
                selection_base_data[alias] = selection_base_data[source.name].copy(
                    deep=False
                )
        insert_index = max(indices) + 1
        sources[insert_index:insert_index] = duplicates

        self.replace_recipe(self.recipe.model_copy(update={"sources": tuple(sources)}))
        self.replace_source_payloads(source_data, selection_base_data)
        return tuple(source.name for source in duplicates)

    def _source_copy_alias(self, source_name: str, reserved: set[str]) -> str:
        stem = f"{source_name}_copy"
        alias = stem
        suffix = 2
        while self.source_alias_error(alias) is not None or alias in reserved:
            alias = f"{stem}_{suffix}"
            suffix += 1
        reserved.add(alias)
        return alias

    def can_move_sources(self, names: Sequence[str], offset: int) -> bool:
        """Return whether any requested source can move by one position."""
        if offset not in {-1, 1}:
            raise ValueError("source move offset must be -1 or 1")
        selected_names = set(names)
        indices = tuple(
            index
            for index, source in enumerate(self.recipe.sources)
            if source.name in selected_names
        )
        if not indices:
            return False
        index_set = set(indices)
        if offset < 0:
            return any(index > 0 and index - 1 not in index_set for index in indices)
        return any(
            index < len(self.recipe.sources) - 1 and index + 1 not in index_set
            for index in indices
        )

    def move_sources(self, names: Sequence[str], offset: int) -> bool:
        """Move requested sources one position while preserving relative order."""
        if offset not in {-1, 1}:
            raise ValueError("source move offset must be -1 or 1")
        selected_names = set(names)
        indices = tuple(
            index
            for index, source in enumerate(self.recipe.sources)
            if source.name in selected_names
        )
        if not indices:
            return False
        sources = list(self.recipe.sources)
        index_set = set(indices)
        moved = False
        if offset < 0:
            for index in indices:
                if index > 0 and index - 1 not in index_set:
                    sources[index - 1], sources[index] = (
                        sources[index],
                        sources[index - 1],
                    )
                    index_set.remove(index)
                    index_set.add(index - 1)
                    moved = True
        else:
            for index in reversed(indices):
                if index < len(sources) - 1 and index + 1 not in index_set:
                    sources[index + 1], sources[index] = (
                        sources[index],
                        sources[index + 1],
                    )
                    index_set.remove(index)
                    index_set.add(index + 1)
                    moved = True
        if not moved:
            return False
        self.replace_recipe(self.recipe.model_copy(update={"sources": tuple(sources)}))
        return True

    def reorder_sources(self, ordered_names: Sequence[str]) -> bool:
        """Reorder sources using an exact permutation of the current names."""
        source_by_name = self.source_by_name()
        names = tuple(ordered_names)
        if len(names) != len(source_by_name) or set(names) != set(source_by_name):
            raise ValueError("source order must be an exact permutation")
        current_order = tuple(source.name for source in self.recipe.sources)
        if names == current_order:
            return False
        self.replace_recipe(
            self.recipe.model_copy(
                update={"sources": tuple(source_by_name[name] for name in names)}
            )
        )
        return True

    def remove_source(self, name: str) -> bool:
        """Remove one unused source."""
        return bool(self.remove_sources((name,)))

    def remove_sources(self, names: Sequence[str]) -> tuple[str, ...]:
        """Remove every requested source that becomes safely removable."""
        pending = list(dict.fromkeys(names))
        sources = list(self.recipe.sources)
        removed: list[str] = []
        while pending:
            next_pending: list[str] = []
            removed_this_pass = False
            for name in pending:
                if not self._source_is_removable(name, sources):
                    next_pending.append(name)
                    continue
                sources = [source for source in sources if source.name != name]
                removed.append(name)
                removed_this_pass = True
            if not removed_this_pass:
                break
            pending = next_pending
        if not removed:
            return ()

        updates: dict[str, typing.Any] = {"sources": tuple(sources)}
        if self.recipe.primary_source in removed:
            updates["primary_source"] = sources[0].name
        source_data = dict(self.source_data)
        selection_base_data = dict(self.source_selection_base_data)
        for name in removed:
            source_data.pop(name, None)
            selection_base_data.pop(name, None)
        self.replace_recipe(self.recipe.model_copy(update=updates))
        self.replace_source_payloads(source_data, selection_base_data)
        return tuple(removed)

    @staticmethod
    def source_data_from_selection(
        data: xr.DataArray,
        source: FigureSourceState,
    ) -> xr.DataArray:
        """Return the public source payload after applying its saved selection."""
        selected = _selected_source_data(data, source)
        return selected.rename(data.name).copy(deep=False)

    def source_selection_input_data(self, source_name: str) -> xr.DataArray | None:
        """Return the raw input to which a source selection applies."""
        data = self.source_selection_base_data.get(source_name)
        if data is not None:
            return data
        source = self.source_by_name().get(source_name)
        selection_source = None if source is None else source.selection_source
        if selection_source is not None and selection_source != source_name:
            data = self.source_data.get(selection_source)
            if data is not None:
                return data
        return self.source_data.get(source_name)

    def update_source_selection_dimension(
        self,
        names: Sequence[str],
        dimension: str,
        mode: str,
        value: typing.Any = None,
        width: typing.Any = None,
    ) -> FigureSourceUpdateResult:
        """Apply one dimension rule to each compatible requested source."""
        if mode not in {"keep", "isel", "qsel", "mean"}:
            raise ValueError(f"unknown source selection mode {mode!r}")
        candidate_data = dict(self.source_data)
        candidate_bases = dict(self.source_selection_base_data)
        candidate_sources = self.source_by_name()
        updated: list[str] = []
        skipped: list[tuple[str, str]] = []
        for source_name in dict.fromkeys(names):
            source = candidate_sources.get(source_name)
            if source is None or source_name not in candidate_data:
                skipped.append((source_name, "source data is unavailable"))
                continue
            try:
                selection = selection_with_dimension(
                    _source_selection(source),
                    dimension,
                    mode,
                    value,
                    width,
                )
                raw_data = candidate_bases.get(source_name)
                if raw_data is None:
                    selection_source = source.selection_source
                    if selection_source is not None and selection_source != source_name:
                        raw_data = candidate_data.get(selection_source)
                    else:
                        raw_data = candidate_data.get(source_name)
                if raw_data is None:
                    skipped.append((source_name, "source data is unavailable"))
                    continue
                updated_source = _source_with_selection(source, selection)
                selected_data = self.source_data_from_selection(
                    raw_data, updated_source
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                skipped.append((source_name, str(exc) or exc.__class__.__name__))
                continue

            trial_sources = dict(candidate_sources)
            trial_sources[source_name] = updated_source
            trial_data = dict(candidate_data)
            trial_data[source_name] = selected_data
            trial_bases = dict(candidate_bases)
            if selection_has_effect(selection):
                trial_bases[source_name] = raw_data
            else:
                trial_bases.pop(source_name, None)
            try:
                trial_data, trial_bases = self.recompute_source_dependents(
                    trial_data,
                    trial_bases,
                    (source_name,),
                    source_by_name=trial_sources,
                )
            except ValueError as exc:
                skipped.append((source_name, str(exc)))
                continue
            candidate_sources = trial_sources
            candidate_data = trial_data
            candidate_bases = trial_bases
            updated.append(source_name)

        result = FigureSourceUpdateResult(
            updated=tuple(updated), skipped=tuple(skipped)
        )
        if not result:
            return result
        self.replace_recipe(
            self.recipe.model_copy(
                update={
                    "sources": tuple(
                        candidate_sources[source.name] for source in self.recipe.sources
                    )
                }
            )
        )
        self.replace_source_payloads(candidate_data, candidate_bases)
        return result

    def recompute_source_dependents(
        self,
        source_data: Mapping[str, xr.DataArray],
        selection_base_data: Mapping[str, xr.DataArray],
        changed_names: Iterable[str],
        *,
        source_by_name: Mapping[str, FigureSourceState] | None = None,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        """Return candidate mappings with every selected descendant refreshed."""
        sources = dict(source_by_name or self.source_by_name())
        candidate_data = dict(source_data)
        candidate_bases = dict(selection_base_data)
        explicit_names = set(changed_names)
        queue: collections.deque[str] = collections.deque(explicit_names)
        refreshed = set(explicit_names)

        while queue:
            parent_name = queue.popleft()
            parent_data = candidate_data.get(parent_name)
            for source in sources.values():
                source_name = source.name
                if (
                    source_name in refreshed
                    or source.selection_source != parent_name
                    or source_name == parent_name
                ):
                    continue
                if parent_data is None:
                    raise ValueError(
                        f"source {source_name!r} depends on unavailable source "
                        f"{parent_name!r}"
                    )
                try:
                    selected_data = self.source_data_from_selection(parent_data, source)
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    message = str(exc) or exc.__class__.__name__
                    raise ValueError(
                        f"could not update dependent source {source_name!r}: {message}"
                    ) from exc
                candidate_data[source_name] = selected_data
                if _source_has_selection(source):
                    candidate_bases[source_name] = parent_data
                else:
                    candidate_bases.pop(source_name, None)
                refreshed.add(source_name)
                queue.append(source_name)
        return candidate_data, candidate_bases

    def _replacement_source_data(
        self,
        alias: str,
        source: FigureSourceState,
        data: xr.DataArray,
        existing_source: FigureSourceState | None,
        *,
        keep_selection_source: bool,
    ) -> tuple[FigureSourceState, xr.DataArray]:
        replacement = self.source_with_name(source, alias)
        if (
            existing_source is not None
            and _source_has_selection(existing_source)
            and not _source_has_selection(replacement)
        ):
            replacement = _source_with_selection(
                replacement.model_copy(
                    update={
                        "selection_source": (
                            existing_source.selection_source
                            if keep_selection_source
                            else None
                        )
                    }
                ),
                _source_selection(existing_source),
            )
        return replacement, self.source_data_from_selection(data, replacement)

    def add_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
    ) -> FigureSourceAddResult:
        """Add sources or refresh matching linked sources atomically per input."""
        existing = self.source_by_name()
        candidate_data = dict(self.source_data)
        candidate_bases = dict(self.source_selection_base_data)
        added: list[tuple[str, str]] = []
        updated: list[tuple[str, str]] = []
        skipped: list[tuple[str, str]] = []
        for incoming_source in sources:
            source = incoming_source
            incoming_name = incoming_source.name
            data = source_data.get(incoming_name)
            if data is None:
                skipped.append(
                    (incoming_name, f"{incoming_name} (source data is unavailable)")
                )
                continue
            linked_matches = (
                [
                    candidate
                    for candidate in existing.values()
                    if candidate.node_uid == source.node_uid
                ]
                if source.node_uid is not None
                else []
            )
            target_name = incoming_name
            existing_source = existing.get(target_name)
            same_linked_source = False
            if linked_matches:
                linked_roots: list[str] = []
                for linked_source in linked_matches:
                    try:
                        root_name = self.source_lineage_names(
                            linked_source.name, existing
                        )[0]
                    except ValueError:
                        continue
                    if root_name not in linked_roots:
                        linked_roots.append(root_name)
                if linked_roots:
                    target_name = (
                        incoming_name
                        if incoming_name in linked_roots
                        else linked_roots[0]
                    )
                    existing_source = existing[target_name]
                    same_linked_source = True
            reserved = set(existing)
            reserved.update(candidate_data)
            if existing_source is not None and not same_linked_source:
                target_name = _source_unique_name(target_name, reserved)
                source = self.source_with_name(source, target_name)
                selected_data = data
            else:
                reserved.add(target_name)
                if existing_source is None:
                    selected_data = data
                else:
                    try:
                        source, selected_data = self._replacement_source_data(
                            target_name,
                            source,
                            data,
                            existing_source,
                            keep_selection_source=True,
                        )
                    except (IndexError, KeyError, TypeError, ValueError) as exc:
                        message = str(exc) or exc.__class__.__name__
                        skipped.append((incoming_name, f"{target_name} ({message})"))
                        continue
            trial_existing = dict(existing)
            trial_existing[target_name] = source
            normalized_sources = self.normalized_source_states(
                tuple(trial_existing.values())
            )
            trial_existing = {
                candidate.name: candidate for candidate in normalized_sources
            }
            trial_data = dict(candidate_data)
            trial_data[target_name] = selected_data
            trial_bases = dict(candidate_bases)
            if same_linked_source and _source_has_selection(source):
                trial_bases[target_name] = data
            else:
                trial_bases.pop(target_name, None)
            try:
                trial_data, trial_bases = self.recompute_source_dependents(
                    trial_data,
                    trial_bases,
                    (target_name,),
                    source_by_name=trial_existing,
                )
            except ValueError as exc:
                skipped.append((incoming_name, f"{target_name} ({exc})"))
                continue
            existing = trial_existing
            candidate_data = trial_data
            candidate_bases = trial_bases
            accepted_entry = (incoming_name, target_name)
            if same_linked_source:
                updated.append(accepted_entry)
            else:
                added.append(accepted_entry)

        result = FigureSourceAddResult(
            added=tuple(added), updated=tuple(updated), skipped=tuple(skipped)
        )
        if not result:
            return result
        self.replace_recipe(
            self.recipe.model_copy(update={"sources": tuple(existing.values())})
        )
        self.replace_source_payloads(candidate_data, candidate_bases)
        return result

    def replace_source(
        self,
        alias: str,
        source: FigureSourceState,
        data: xr.DataArray,
    ) -> FigureSourceUpdateResult:
        """Replace source data while preserving its recipe-facing alias."""
        source_list = list(self.recipe.sources)
        existing_source: FigureSourceState | None = None
        existing_index: int | None = None
        for candidate_index, candidate_source in enumerate(source_list):
            if candidate_source.name == alias:
                existing_source = candidate_source
                existing_index = candidate_index
                break
        else:
            if alias not in self.source_data:
                return FigureSourceUpdateResult(
                    skipped=((alias, "source data is unavailable"),)
                )
            source_list.append(self.source_with_name(source, alias))
            existing_index = len(source_list) - 1

        replacement_name = alias
        preserve_selection_parent = False
        if (
            existing_source is not None
            and existing_source.node_uid is not None
            and existing_source.node_uid == source.node_uid
        ):
            source_by_name = {candidate.name: candidate for candidate in source_list}
            try:
                root_name = self.source_lineage_names(alias, source_by_name)[0]
            except ValueError:
                root_name = alias
            root_source = source_by_name[root_name]
            if root_source.selection_source is None:
                replacement_name = root_name
                existing_source = root_source
                existing_index = next(
                    index
                    for index, candidate in enumerate(source_list)
                    if candidate.name == root_name
                )
                preserve_selection_parent = True
        try:
            replacement, selected_data = self._replacement_source_data(
                replacement_name,
                source,
                data,
                existing_source,
                keep_selection_source=preserve_selection_parent,
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            return FigureSourceUpdateResult(
                skipped=((alias, str(exc) or exc.__class__.__name__),)
            )
        source_list[existing_index] = replacement
        source_list = list(self.normalized_source_states(source_list))

        candidate_data = dict(self.source_data)
        candidate_data[replacement_name] = selected_data
        candidate_bases = dict(self.source_selection_base_data)
        if _source_has_selection(replacement):
            candidate_bases[replacement_name] = data
        else:
            candidate_bases.pop(replacement_name, None)
        source_by_name = {candidate.name: candidate for candidate in source_list}
        try:
            candidate_data, candidate_bases = self.recompute_source_dependents(
                candidate_data,
                candidate_bases,
                (replacement_name,),
                source_by_name=source_by_name,
            )
        except ValueError as exc:
            return FigureSourceUpdateResult(skipped=((alias, str(exc)),))

        self.replace_recipe(
            self.recipe.model_copy(update={"sources": tuple(source_list)})
        )
        self.replace_source_payloads(candidate_data, candidate_bases)
        return FigureSourceUpdateResult(updated=(replacement_name,))

    def refresh_sources(
        self, source_data: Mapping[str, xr.DataArray]
    ) -> FigureSourceUpdateResult:
        """Refresh source payloads atomically per requested source."""
        source_by_name = self.source_by_name()
        candidate_data = dict(self.source_data)
        candidate_bases = dict(self.source_selection_base_data)
        updated: list[str] = []
        skipped: list[tuple[str, str]] = []
        for source_name, data in source_data.items():
            source = source_by_name.get(source_name)
            if source is None:
                selected_data = data
            else:
                try:
                    selected_data = self.source_data_from_selection(data, source)
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    skipped.append((source_name, str(exc) or exc.__class__.__name__))
                    continue
            trial_data = dict(candidate_data)
            trial_data[source_name] = selected_data
            trial_bases = dict(candidate_bases)
            if source is not None and _source_has_selection(source):
                trial_bases[source_name] = data
            else:
                trial_bases.pop(source_name, None)
            try:
                trial_data, trial_bases = self.recompute_source_dependents(
                    trial_data,
                    trial_bases,
                    (source_name,),
                    source_by_name=source_by_name,
                )
            except ValueError as exc:
                skipped.append((source_name, str(exc)))
                continue
            candidate_data = trial_data
            candidate_bases = trial_bases
            updated.append(source_name)
        result = FigureSourceUpdateResult(
            updated=tuple(updated), skipped=tuple(skipped)
        )
        if not result:
            return result
        self.replace_source_payloads(candidate_data, candidate_bases)
        return result

    def discard_source_data(self, names: Iterable[str]) -> tuple[str, ...]:
        """Discard backing data for recipe sources that became unavailable."""
        recipe_names = {source.name for source in self.recipe.sources}
        discarded = tuple(
            name
            for name in dict.fromkeys(names)
            if name in recipe_names
            and (name in self.source_data or name in self.source_selection_base_data)
        )
        if not discarded:
            return ()
        source_data = dict(self.source_data)
        selection_base_data = dict(self.source_selection_base_data)
        for name in discarded:
            source_data.pop(name, None)
            selection_base_data.pop(name, None)
        self.replace_source_payloads(source_data, selection_base_data)
        return discarded

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
