"""Recorded data dependencies and live input bindings for ImageTool."""

from __future__ import annotations

import copy
import threading
import typing
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from graphlib import TopologicalSorter

import xarray as xr

from erlab._provenance._records import (
    Application,
    Binding,
    DataRevision,
    EarlierStep,
    FileLoad,
    FileSource,
    Follow,
    ImageToolEntry,
    PendingRecipe,
    Pin,
    Recipe,
    check_id,
    checkpoint_data,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from erlab.interactive.imagetool._provenance._model import FileReplayCall


@dataclass(frozen=True, eq=False)
class _StoredRevision:
    record: DataRevision
    recipe: Recipe | FileLoad | None
    checkpoint: xr.DataArray | None
    # Keep exact inputs and earlier steps for as long as this revision is reachable.
    parents: tuple[_StoredRevision, ...]


@dataclass
class _Evaluation:
    """Values shared within one request. Nothing is published during evaluation."""

    values: dict[Binding, tuple[str, xr.DataArray]] = field(default_factory=dict)
    sources: dict[str, _StoredRevision] = field(default_factory=dict)
    revisions: dict[str, _StoredRevision] = field(default_factory=dict)
    entries: dict[str, tuple[str, xr.DataArray]] = field(default_factory=dict)


class ProvenanceGraph:
    """A single-owner, synchronous graph with stable ImageTool entries.

    Recorded edges always refer to existing nodes. Publishing a fresh revision can
    thus only extend the recorded DAG. The separate index of entry followers tracks
    which results can become stale when an entry moves. Entries, recipes, evaluation
    requests and explicit history retention keep their revisions reachable. Other
    revisions and their payloads can be released.
    """

    def __init__(
        self,
        *,
        read_file: Callable[[FileSource, FileReplayCall], xr.DataArray] | None = None,
    ) -> None:
        self._read_file = read_file
        self._owner = threading.get_ident()
        self._files: dict[str, FileSource] = {}
        self._entries: dict[str, ImageToolEntry] = {}
        self._revisions: weakref.WeakValueDictionary[str, _StoredRevision] = (
            weakref.WeakValueDictionary()
        )
        self._entry_roots: dict[str, tuple[_StoredRevision, ...]] = {}
        self._displayed: dict[str, xr.DataArray] = {}
        self._followers: dict[str, set[str]] = {}
        self._stale: set[str] = set()
        self._generation = 0

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def revision_ids(self) -> frozenset[str]:
        return frozenset(self._revisions)

    def entry(self, entry_id: str) -> ImageToolEntry:
        # A frozen typed operation can still contain mutable parameter values.
        return copy.deepcopy(self._entries[entry_id])

    def revision(self, revision_id: str) -> DataRevision:
        return copy.deepcopy(self._revisions[revision_id].record)

    @contextmanager
    def retain(self, revision_id: str) -> Iterator[None]:
        """Keep a selected revision and its inputs available across entry updates."""
        saved = self._revisions[revision_id]
        try:
            yield
        finally:
            del saved

    def is_virtual(self, revision_id: str) -> bool:
        if revision_id not in self._revisions:
            raise KeyError(revision_id)
        return revision_id not in self._displayed

    def is_stale(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            raise KeyError(entry_id)
        return entry_id in self._stale

    def next_inputs(
        self, entry_id: str
    ) -> tuple[tuple[str, str | Follow | EarlierStep], ...]:
        """Return node IDs, or bindings whose next revision has yet to be produced."""
        recipe = self._entries[entry_id].recipe
        if recipe is None:
            return ()
        if isinstance(recipe, FileLoad):
            return recipe.inputs
        inputs: list[tuple[str, str | Follow | EarlierStep]] = []
        for slot, binding in recipe.inputs:
            if isinstance(binding, Follow):
                source = self._entries[binding.entry_id]
                inputs.append(
                    (
                        slot,
                        binding
                        if binding.entry_id in self._stale
                        else source.revision_id,
                    )
                )
            elif isinstance(binding, Pin):
                inputs.append((slot, binding.revision_id))
            else:
                inputs.append((slot, copy.deepcopy(binding)))
        return tuple(inputs)

    def data(self, revision_id: str) -> xr.DataArray:
        """Read a revision, replaying virtual steps from their recorded inputs."""
        root = Pin(revision_id)
        result = self._evaluate(root)
        return result.values[root][1].copy(deep=True)

    def _check_new_entry(self, entry_id: str, parent_id: str | None) -> None:
        check_id(entry_id)
        if entry_id in self._entries:
            raise ValueError(f"ImageTool entry {entry_id!r} already exists")
        if parent_id is not None and parent_id not in self._entries:
            raise KeyError(parent_id)

    def add_source(
        self,
        entry_id: str,
        data: xr.DataArray,
        *,
        auto: bool = False,
        parent_id: str | None = None,
    ) -> str:
        self._check_new_entry(entry_id, parent_id)
        generation = self._write_generation()
        owned = checkpoint_data(data)
        self._check_generation(generation)
        revision_id = uuid.uuid4().hex
        stored = _StoredRevision(DataRevision(revision_id, None, True), None, owned, ())
        self._revisions[revision_id] = stored
        self._entry_roots[entry_id] = (stored,)
        self._displayed[revision_id] = owned
        self._entries[entry_id] = ImageToolEntry(
            entry_id, revision_id, None, auto, parent_id, len(self._entries)
        )
        self._generation += 1
        return revision_id

    def file_source(self, path: str) -> FileSource:
        self._write_generation()
        if not path:
            raise ValueError("A file source needs a path")
        if path not in self._files:
            self._files[path] = FileSource(uuid.uuid4().hex, path)
        return self._files[path]

    def derive(
        self,
        entry_id: str,
        recipe: Recipe | FileLoad,
        *,
        auto: bool = False,
        parent_id: str | None = None,
    ) -> str:
        self._check_new_entry(entry_id, parent_id)
        return self._run(
            entry_id, copy.deepcopy(recipe), auto=auto, parent_id=parent_id
        )

    def replace(self, entry_id: str, recipe: Recipe | FileLoad) -> str:
        current = self._entries[entry_id]
        recipe = copy.deepcopy(recipe)
        if isinstance(recipe, Recipe):
            # Reading the target consumes its displayed value. On the next refresh,
            # replay its previous producer, or pin an irreplaceable source.
            prior: EarlierStep | Pin
            if current.recipe is None:
                prior = Pin(current.revision_id)
            elif current.recipe is self._revisions[current.revision_id].recipe:
                prior = EarlierStep(current.revision_id)
            else:
                prior = EarlierStep(current.revision_id, PendingRecipe(current.recipe))
            recipe = replace(
                recipe,
                inputs=tuple(
                    (slot, prior if binding == Follow(entry_id) else binding)
                    for slot, binding in recipe.inputs
                ),
            )
        return self._run(entry_id, recipe)

    def set_auto(self, entry_id: str, *, enabled: bool) -> None:
        self._write_generation()
        self._entries[entry_id] = replace(self._entries[entry_id], auto=enabled)
        self._generation += 1

    def set_recipe(self, entry_id: str, recipe: Recipe | FileLoad) -> None:
        """Change the next calculation without changing the displayed revision."""
        self._write_generation()
        current = self._entries[entry_id]
        recipe = copy.deepcopy(recipe)
        sources = self._check_recipe(entry_id, recipe)
        self._index_sources(entry_id, current.recipe, sources)
        self._entry_roots[entry_id] = (
            self._revisions[current.revision_id],
            *(self._revisions[node] for node in self._recipe_revisions(recipe)),
        )
        self._entries[entry_id] = replace(current, recipe=recipe)
        self._stale.add(entry_id)
        self._stale.update(self._descendants([entry_id]))
        self._generation += 1

    def refresh(self, entry_id: str) -> str:
        entry = self._entries[entry_id]
        if entry.recipe is None:
            return entry.revision_id
        return self._run(entry_id, entry.recipe, refresh=True)

    def _sources(self, recipe: Recipe | FileLoad | None) -> set[str]:
        """Find entries followed directly or through an earlier producing step."""
        pending = [recipe]
        sources: set[str] = set()
        visited: set[EarlierStep] = set()
        while pending:
            step = pending.pop()
            if isinstance(step, FileLoad):
                if self._files.get(step.source.path) != step.source:
                    raise ValueError("Unknown file source")
            elif isinstance(step, Recipe):
                for _, binding in step.inputs:
                    if isinstance(binding, Follow):
                        if binding.entry_id not in self._entries:
                            raise KeyError(binding.entry_id)
                        sources.add(binding.entry_id)
                    elif isinstance(binding, Pin):
                        if binding.revision_id not in self._revisions:
                            raise KeyError(binding.revision_id)
                    elif isinstance(binding, EarlierStep):
                        if binding not in visited:
                            pending.append(self._earlier_recipe(binding))
                            visited.add(binding)
                    else:
                        raise TypeError(f"Invalid refresh binding {binding!r}")
        return sources

    @staticmethod
    def _recipe_revisions(recipe: Recipe | FileLoad) -> set[str]:
        nodes: set[str] = set()
        pending = [recipe]
        edits: set[PendingRecipe] = set()
        while pending:
            step = pending.pop()
            if isinstance(step, Recipe):
                for _, binding in step.inputs:
                    if isinstance(binding, Pin | EarlierStep):
                        nodes.add(binding.revision_id)
                    if (
                        isinstance(binding, EarlierStep)
                        and binding.pending is not None
                        and binding.pending not in edits
                    ):
                        edits.add(binding.pending)
                        pending.append(binding.pending.recipe)
        return nodes

    def _earlier_recipe(self, binding: EarlierStep) -> Recipe | FileLoad:
        stored = self._revisions[binding.revision_id]
        if binding.pending is not None:
            return binding.pending.recipe
        recipe = stored.recipe
        if recipe is None:
            raise ValueError("A source revision has no earlier producing step")
        return recipe

    def _earlier_input_changed(self, recipe: Recipe | FileLoad) -> bool:
        """Check a consumed earlier step against the inputs it would use now."""
        if isinstance(recipe, FileLoad):
            return False
        pending = [
            (binding, binding.revision_id)
            for _, binding in recipe.inputs
            if isinstance(binding, EarlierStep)
        ]
        visited: set[str] = set()
        while pending:
            binding, node = pending.pop()
            if binding.pending is not None and binding.revision_id == node:
                return True
            if node in visited:
                continue
            visited.add(node)
            stored = self._revisions[node]
            if not isinstance(stored.recipe, Recipe):
                continue
            applied = typing.cast("Application", stored.record.producer)
            for (_, source), (_, used) in zip(
                stored.recipe.inputs, applied.inputs, strict=True
            ):
                if (
                    isinstance(source, Follow)
                    and self._entries[source.entry_id].revision_id != used
                ):
                    return True
                if isinstance(source, EarlierStep):
                    pending.append((source, used))
        return False

    def _descendants(self, entry_ids: Iterable[str]) -> set[str]:
        pending = list(entry_ids)
        visited = set(pending)
        descendants: set[str] = set()
        while pending:
            for child in self._followers.get(pending.pop(), ()):
                descendants.add(child)
                if child not in visited:
                    visited.add(child)
                    pending.append(child)
        return descendants

    def _check_recipe(self, entry_id: str, recipe: object) -> set[str]:
        if not isinstance(recipe, Recipe | FileLoad):
            raise TypeError("Expected an operation recipe or a file load")
        sources = self._sources(recipe)
        if sources and (
            entry_id in sources or not sources.isdisjoint(self._descendants([entry_id]))
        ):
            raise ValueError("Refresh bindings would create a cycle")
        return sources

    def _index_sources(
        self, entry_id: str, old: Recipe | FileLoad | None, sources: set[str]
    ) -> None:
        for source in self._sources(old):
            self._followers[source].remove(entry_id)
        for source in sources:
            self._followers.setdefault(source, set()).add(entry_id)

    def _check_generation(self, generation: int) -> None:
        if self._generation != generation:
            raise RuntimeError("Provenance changed during evaluation")

    def _write_generation(self) -> int:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Only the graph's owner thread can publish changes")
        return self._generation

    def _run(
        self,
        entry_id: str,
        recipe: Recipe | FileLoad,
        *,
        refresh: bool = False,
        auto: bool = False,
        parent_id: str | None = None,
    ) -> str:
        generation = self._write_generation()
        sources = (
            self._sources(recipe) if refresh else self._check_recipe(entry_id, recipe)
        )
        root = Follow(entry_id)
        result = self._evaluate(root, recipe=recipe, refresh=refresh)
        self._check_generation(generation)

        # Commit the prepared values and edges together after every operation succeeds.
        self._revisions.update(result.revisions)
        for target, (revision_id, data) in result.entries.items():
            old = self._entries.get(target)
            stored = result.revisions[revision_id]
            effective = stored.recipe
            if old is None:
                entry = ImageToolEntry(
                    target, revision_id, effective, auto, parent_id, len(self._entries)
                )
            else:
                del self._displayed[old.revision_id]
                entry = replace(old, revision_id=revision_id, recipe=effective)
            if target == entry_id and not refresh:
                self._index_sources(target, old.recipe if old else None, sources)
            self._entries[target] = entry
            self._displayed[revision_id] = data
            self._entry_roots[target] = (stored,)

        self._stale.update(self._descendants(result.entries))
        self._stale.difference_update(result.entries)
        if not refresh and (
            sources & self._stale or self._earlier_input_changed(recipe)
        ):
            # Derive and Replace consume the data on screen, including stale data.
            self._stale.add(entry_id)
        self._generation += 1
        return result.values[root][0]

    def _plan_input(
        self, binding: Binding, *, refresh: bool
    ) -> Recipe | FileLoad | Pin | tuple[str, xr.DataArray]:
        if isinstance(binding, Follow):
            entry = self._entries[binding.entry_id]
            if refresh and binding.entry_id in self._stale and entry.recipe is not None:
                return entry.recipe
            return Pin(entry.revision_id)
        if isinstance(binding, EarlierStep):
            if refresh:
                return self._earlier_recipe(binding)
            return Pin(binding.revision_id)

        stored = self._revisions[binding.revision_id]
        data = self._displayed.get(binding.revision_id, stored.checkpoint)
        if data is not None:
            return binding.revision_id, data
        producer = stored.record.producer
        if not isinstance(producer, Application):
            raise TypeError(f"Revision {binding.revision_id!r} needs a checkpoint")
        return Recipe(
            producer.operation,
            tuple((slot, Pin(node)) for slot, node in producer.inputs),
        )

    def _evaluate(
        self,
        root: Binding,
        *,
        recipe: Recipe | FileLoad | None = None,
        refresh: bool = False,
    ) -> _Evaluation:
        plan: dict[Binding, Recipe | FileLoad | Pin | tuple[str, xr.DataArray]] = {}
        order: TopologicalSorter[Binding] = TopologicalSorter()
        result = _Evaluation()
        if recipe is not None:
            # Pending edits can pin inputs that Replace does not evaluate.
            result.sources.update(
                (node, self._revisions[node]) for node in self._recipe_revisions(recipe)
            )
        pending = [root]
        while pending:
            binding = pending.pop()
            if binding in plan:
                continue
            if isinstance(binding, Pin | EarlierStep):
                result.sources[binding.revision_id] = self._revisions[
                    binding.revision_id
                ]
            step = (
                recipe
                if binding == root and recipe is not None
                else self._plan_input(binding, refresh=refresh)
            )
            plan[binding] = step
            dependencies: tuple[Binding, ...] = ()
            if isinstance(step, Pin):
                dependencies = (step,)
            elif isinstance(step, Recipe):
                dependencies = tuple(source for _, source in step.inputs)
            order.add(binding, *dependencies)
            pending.extend(dependencies)

        for binding in order.static_order():
            step = plan[binding]
            if isinstance(step, tuple):
                result.values[binding] = step
            elif isinstance(step, Pin):
                result.values[binding] = result.values[step]
            else:
                self._evaluate_step(result, binding, step)
        return result

    def _evaluate_step(
        self, result: _Evaluation, binding: Binding, step: Recipe | FileLoad
    ) -> None:
        producer: Application | FileLoad
        if isinstance(step, FileLoad):
            if self._read_file is None:
                raise ValueError(f"No file reader for {step.source.path!r}")
            data = checkpoint_data(
                self._read_file(step.source, copy.deepcopy(step.call))
            )
            producer = step
        else:
            inputs = tuple(
                (slot, result.values[source][0]) for slot, source in step.inputs
            )
            producer = Application(step.operation, inputs)
            values = dict(result.values[source] for _, source in step.inputs)
            data = producer.apply(
                {node: value.copy(deep=True) for node, value in values.items()}
            )
            if not isinstance(data, xr.DataArray):
                raise TypeError("An operation must produce a DataArray")
        if isinstance(binding, Pin):
            result.values[binding] = (binding.revision_id, data)
            return
        revision_id = uuid.uuid4().hex
        result.values[binding] = (revision_id, data)
        parent_ids = self._recipe_revisions(step)
        if isinstance(producer, Application):
            parent_ids.update(node for _, node in producer.inputs)
        result.revisions[revision_id] = _StoredRevision(
            DataRevision(revision_id, producer, isinstance(step, FileLoad)),
            step,
            data if isinstance(step, FileLoad) else None,
            tuple(
                result.revisions.get(node) or result.sources[node]
                for node in parent_ids
            ),
        )
        if isinstance(binding, Follow):
            result.entries[binding.entry_id] = (revision_id, data)
