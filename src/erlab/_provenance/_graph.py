"""Recorded data dependencies and live input bindings for ImageTool."""

from __future__ import annotations

import copy
import struct
import threading
import typing
import uuid
import weakref
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from graphlib import TopologicalSorter

import pydantic
import xarray as xr

from erlab._provenance._checkpoint import (
    FileVersion,
    checkpoint_data,
    editable_data,
    protected_data,
)
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
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence

    from erlab._provenance._model import (
        FileReplayCall,
        ScriptInputDataRole,
        ToolProvenanceOperation,
    )


@dataclass(eq=False, slots=True, weakref_slot=True)
class _StoredRevision:
    """A record staged in dependency order and sealed at publication."""

    record: DataRevision
    # Re-evaluations of one producing step keep its original identity.
    producer_id: str
    recipe: Recipe | FileLoad | None
    checkpoint: xr.DataArray | None
    # Keep exact inputs and earlier steps for as long as this revision is reachable.
    parents: tuple[_StoredRevision, ...]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _VisibleRevision:
    """Keep the recorded revision and its current array together while on screen."""

    stored: _StoredRevision
    data: xr.DataArray


_Mode: typing.TypeAlias = typing.Literal["displayed", "refresh", "reload"]
_Action: typing.TypeAlias = Recipe | FileLoad | Binding | tuple[str, xr.DataArray]


class _SupersededEvaluation(RuntimeError):
    """An observed entry changed before the result could be published."""


def _setting_key(value: typing.Any) -> Hashable:
    """Compare declared settings without computing arrays or invoking user equality."""
    cls = type(value)
    if cls in (str, bytes, int, bool, type(None)):
        return cls, value
    if cls is float:
        return cls, struct.pack("!d", value)
    if cls in (list, tuple):
        return cls, tuple(_setting_key(item) for item in value)
    if cls is dict:
        return cls, tuple((_setting_key(k), _setting_key(v)) for k, v in value.items())
    if cls is slice:
        return cls, _setting_key((value.start, value.stop, value.step))
    if isinstance(value, pydantic.BaseModel) and not (
        value.model_extra or value.__pydantic_private__
    ):
        return cls, tuple(
            _setting_key(getattr(value, name)) for name in type(value).model_fields
        )
    # Opaque parameters require the very same snapshot to establish equivalence.
    return cls, id(value)


def _share_producer(
    bindings: list[Follow | EarlierStep],
    actions: dict[Binding, _Action],
    dependencies: dict[Binding, tuple[Binding, ...]],
) -> None:
    """Make equivalent requests for one producer depend on a single calculation."""
    if len(bindings) < 2:
        return
    # Copies of an unedited recipe retain the captured settings and bindings.
    snapshots: dict[tuple[int, FileSource | None], list[Follow | EarlierStep]] = {}
    for binding in bindings:
        step = typing.cast("Recipe | FileLoad", actions[binding])
        source = step.source if isinstance(step, FileLoad) else None
        snapshots.setdefault((id(step._captured), source), []).append(binding)

    versions: dict[Hashable, Follow | EarlierStep] = {}
    for group in snapshots.values():
        representative = group[0]
        if len(snapshots) > 1:
            # Compare settings once per distinct snapshot of this producer.
            step = typing.cast("Recipe | FileLoad", actions[representative])
            key: Hashable
            if isinstance(step, FileLoad):
                key = (step.source, _setting_key(step._captured))
            else:
                key = (
                    step._captured.inputs,
                    step._captured.execution,
                    _setting_key(step._captured.operation),
                )
            representative = versions.setdefault(key, representative)
        for binding in group:
            if representative != binding:
                actions[binding] = representative
                dependencies[binding] = (representative,)


@dataclass(frozen=True)
class UpdatePlan:
    """Planned reads and calculations, and followers affected by their publication."""

    entries: tuple[str, ...]
    earlier_steps: tuple[EarlierStep, ...]
    file_loads: tuple[FileLoad, ...]
    current_inputs: tuple[tuple[str, str], ...]
    pinned_inputs: tuple[str, ...]
    affected_followers: frozenset[str]


class MissingFileSource(FileNotFoundError):
    def __init__(
        self,
        source: FileSource,
        displayed: ImageToolEntry | None,
        cause: FileNotFoundError,
    ) -> None:
        self.source = source
        self.displayed_revision_id = displayed.revision_id if displayed else None
        message = f"Could not read file source {source.path!r}: {cause}"
        if displayed is not None:
            message += (
                f". The displayed revision {displayed.revision_id!r} of "
                f"{displayed.entry_id!r} and its source checkpoints remain available."
            )
        super().__init__(message)


@dataclass
class _Plan:
    root: Binding
    steps: tuple[tuple[Binding, _Action], ...]
    sources: dict[str, _StoredRevision]
    observed: dict[str, tuple[ImageToolEntry | None, bool]]
    producer_ids: dict[Binding, str]
    display_sources: dict[str, EarlierStep]

    def describe(self, graph: ProvenanceGraph) -> UpdatePlan:
        entries = []
        earlier = []
        files = []
        current = []
        pins: dict[str, None] = {}
        for binding, step in self.steps:
            if isinstance(binding, Pin):
                # Reconstructing a pin only uses recorded data.
                continue
            if isinstance(step, Recipe | FileLoad | Follow | EarlierStep):
                if isinstance(binding, Follow):
                    entries.append(binding.entry_id)
                else:
                    earlier.append(copy.deepcopy(binding))
                if isinstance(step, FileLoad):
                    files.append(copy.deepcopy(step))
                elif isinstance(step, Recipe):
                    for _, source in step._captured.inputs:
                        if isinstance(source, Pin):
                            pins[source.revision_id] = None
            elif isinstance(binding, Follow) and isinstance(step, Pin):
                current.append((binding.entry_id, step.revision_id))
        return UpdatePlan(
            entries=tuple(entries),
            earlier_steps=tuple(earlier),
            file_loads=tuple(files),
            current_inputs=tuple(current),
            pinned_inputs=tuple(pins),
            affected_followers=frozenset(graph._descendants(entries)).difference(
                entries
            ),
        )


def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise CancelledError("Provenance evaluation was cancelled")


@dataclass
class _Evaluation:
    """Values shared within one request. Nothing is published during evaluation."""

    values: dict[Binding, tuple[str, xr.DataArray]] = field(default_factory=dict)
    sources: dict[str, _StoredRevision] = field(default_factory=dict)
    revisions: dict[str, _StoredRevision] = field(default_factory=dict)
    entries: dict[str, _EntryResult] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatedDisplay:
    """An accepted filter evaluated on the pending source revision."""

    revision: DataRevision
    data: xr.DataArray


@dataclass(frozen=True)
class EvaluatedEntry:
    """One pending display update, available to the host before graph publication."""

    entry_id: str
    revision: DataRevision
    recipe: Recipe | FileLoad
    data: xr.DataArray
    display: EvaluatedDisplay | None = None


@dataclass(frozen=True)
class _EntryResult:
    """Both roles in one pending entry update. They coincide without a filter."""

    source: tuple[str, xr.DataArray]
    displayed: tuple[str, xr.DataArray]

    def for_host(
        self, entry_id: str, revisions: dict[str, _StoredRevision]
    ) -> EvaluatedEntry:
        source_id, source_data = self.source
        shown_id, shown_data = self.displayed
        source = revisions[source_id]
        return EvaluatedEntry(
            entry_id,
            copy.deepcopy(source.record),
            copy.deepcopy(typing.cast("Recipe | FileLoad", source.recipe)),
            source_data,
            None
            if shown_id == source_id
            else EvaluatedDisplay(
                copy.deepcopy(revisions[shown_id].record), shown_data
            ),
        )


class ProvenanceGraph:
    """A single-owner, synchronous graph with stable ImageTool entries.

    Recorded edges always refer to existing nodes. Publishing a fresh revision can
    thus only extend the recorded DAG. The separate index of entry followers tracks
    which results can become stale when an entry moves. Entries, recipes, evaluation
    requests and explicit history retention keep their revisions reachable. Other
    revisions and their payloads can be released.

    Auto drains eligible work on the owner thread. A host can supply
    ``schedule_auto`` to defer the drain (for example, to the next event loop turn)
    and coalesce several changes. Automatic failures are available via ``auto_error``.
    """

    def __init__(
        self,
        *,
        read_file: Callable[
            [FileSource, FileReplayCall],
            xr.DataArray | tuple[xr.DataArray, FileVersion],
        ]
        | None = None,
        schedule_auto: Callable[[Callable[[], None]], None] | None = None,
        install: Callable[[tuple[EvaluatedEntry, ...]], AbstractContextManager[None]]
        | None = None,
    ) -> None:
        self._read_file = read_file
        self._install = install
        self._owner = threading.get_ident()
        self._files: weakref.WeakValueDictionary[str, FileSource] = (
            weakref.WeakValueDictionary()
        )
        self._entries: dict[str, ImageToolEntry] = {}
        self._closed: dict[str, _StoredRevision] = {}
        self._revisions: weakref.WeakValueDictionary[str, _StoredRevision] = (
            weakref.WeakValueDictionary()
        )
        self._entry_roots: dict[
            str, tuple[_VisibleRevision | _StoredRevision, ...]
        ] = {}
        self._displayed: weakref.WeakValueDictionary[str, _VisibleRevision] = (
            weakref.WeakValueDictionary()
        )
        self._followers: dict[str, set[str]] = {}
        self._stale: set[str] = set()
        self._generation = 0
        self._schedule_auto = schedule_auto
        self._auto_pending: dict[str, None] = {}
        self._auto_scheduled = False
        self._posting_auto = False
        self._auto_errors: dict[str, str] = {}

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def revision_ids(self) -> frozenset[str]:
        return frozenset(self._revisions)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def current_revision(self, entry_id: str) -> str:
        return self._entries[entry_id].revision_id

    def source_revision(self, entry_id: str) -> str:
        return self._entries[entry_id].source_revision_id

    def followed_entries(self, entry_id: str) -> set[str]:
        return self._sources(self._entries[entry_id].recipe)

    def has_followers(self, entry_id: str) -> bool:
        return bool(self._followers.get(entry_id))

    def has_recipe(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        return entry is not None and entry.recipe is not None

    def has_recorded_derivation(self, entry_id: str) -> bool:
        """Check whether the displayed revision has a recorded producer."""
        entry = self._entries.get(entry_id)
        return (
            entry is not None
            and self._revisions[entry.revision_id].record.producer is not None
        )

    def entry(self, entry_id: str) -> ImageToolEntry:
        # A frozen typed operation can still contain mutable parameter values.
        return copy.deepcopy(self._entries[entry_id])

    def revision(self, revision_id: str) -> DataRevision:
        return copy.deepcopy(self._revisions[revision_id].record)

    def on_release(self, revision_id: str, release: Callable[[str], None]) -> None:
        """Release an exported projection when the last graph owner goes away."""
        finalizer = weakref.finalize(self._revisions[revision_id], release, revision_id)
        finalizer.atexit = False

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
        for slot, binding in recipe._captured.inputs:
            if isinstance(binding, Follow):
                if binding.entry_id in self._closed:
                    inputs.append(
                        (slot, self._closed[binding.entry_id].record.revision_id)
                    )
                    continue
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
        result = self._evaluate(self._plan(root))
        return editable_data(result.values[root][1])

    def recorded_lineage(self, revision_id: str) -> tuple[DataRevision, ...]:
        """Return each recorded ancestor once, before the revision that used it."""
        order: TopologicalSorter[str] = TopologicalSorter()
        pending = [revision_id]
        seen: set[str] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            producer = self._revisions[node].record.producer
            inputs = (
                tuple(source for _, source in producer.inputs)
                if isinstance(producer, Application)
                else ()
            )
            order.add(node, *inputs)
            pending.extend(reversed(inputs))
        return tuple(self.revision(node) for node in order.static_order())

    def _check_new_entry(self, entry_id: str, parent_id: str | None) -> None:
        check_id(entry_id)
        if entry_id in self._entries or entry_id in self._closed:
            raise ValueError(f"ImageTool entry {entry_id!r} already exists")
        if parent_id is not None and parent_id not in self._entries:
            raise KeyError(parent_id)

    def add_source(
        self,
        entry_id: str,
        data: xr.DataArray,
        *,
        backing: FileVersion | None = None,
        auto: bool = False,
        parent_id: str | None = None,
    ) -> str:
        self._check_new_entry(entry_id, parent_id)
        generation = self._write_generation()
        owned = checkpoint_data(data, backing=backing)
        if self._generation != generation:
            self._check_new_entry(entry_id, parent_id)
        revision_id = uuid.uuid4().hex
        stored = _StoredRevision(
            DataRevision(revision_id, None, True, backing), revision_id, None, owned, ()
        )
        self._revisions[revision_id] = stored
        entry = ImageToolEntry(
            entry_id,
            revision_id,
            None,
            auto,
            parent_id,
            len(self._entries),
            revision_id,
        )
        self._bind_entry(entry, owned, owned)
        self._generation += 1
        return revision_id

    def adopt_checkpoint(
        self,
        entry_id: str,
        revision_id: str,
        owned: xr.DataArray,
        *,
        file_load: FileLoad | None = None,
        backing: FileVersion | None = None,
        display: tuple[str, ToolProvenanceOperation, xr.DataArray] | None = None,
    ) -> None:
        """Bind an already protected checkpoint from the host revision store."""
        self._write_generation()
        check_id(entry_id)
        check_id(revision_id)
        if not isinstance(owned, xr.DataArray):
            raise TypeError("A checkpoint must be a DataArray")
        current = self._entries.get(entry_id)
        if (
            current is not None
            and current.source_revision_id == revision_id
            and current.revision_id == (revision_id if display is None else display[0])
        ):
            return
        self._publish_adopted(
            entry_id,
            _StoredRevision(
                DataRevision(revision_id, file_load, True, backing),
                revision_id,
                file_load,
                owned,
                (),
            ),
            owned,
            display=display,
        )

    def adopt_application(
        self,
        entry_id: str,
        revision_id: str,
        owned: xr.DataArray,
        operation: ToolProvenanceOperation,
        *,
        input_role: ScriptInputDataRole,
        display: tuple[str, ToolProvenanceOperation, xr.DataArray] | None = None,
    ) -> None:
        """Record an edit of the selected role and publish its source and display."""
        self._write_generation()
        check_id(revision_id)
        if not isinstance(owned, xr.DataArray):
            raise TypeError("An application result must be a DataArray")
        entry = self._entries[entry_id]
        binding = self._previous_input(entry, input_role)
        if binding.revision_id == revision_id:
            return
        stored = self._record_unary(
            self._revisions[binding.revision_id], revision_id, operation, binding
        )
        self._publish_adopted(
            entry_id,
            stored,
            owned,
            display=display,
            auto=entry.auto,
            stale=entry_id in self._stale,
        )

    def _record_unary(
        self,
        parent: _StoredRevision,
        revision_id: str,
        operation: ToolProvenanceOperation,
        binding: Binding,
    ) -> _StoredRevision:
        check_id(revision_id)
        recipe = Recipe.bind(operation, binding)
        application = Application(
            recipe._captured.operation,
            (("data", parent.record.revision_id),),
            recipe._captured.execution,
        )
        # Future inputs of an edited earlier step can differ from every recorded
        # operand. A later edit can consume this revision after its entry moves.
        parents: tuple[_StoredRevision, ...] = (parent,)
        if isinstance(binding, EarlierStep) and binding.pending is not None:
            parents = (
                parent,
                *(
                    self._revisions[key]
                    for key in self._recipe_revisions(binding.pending.recipe)
                ),
            )
        return _StoredRevision(
            DataRevision(revision_id, application, False),
            revision_id,
            recipe,
            None,
            parents,
        )

    def adopt_file_view(
        self,
        entry_id: str,
        file_load: FileLoad,
        backing: FileVersion,
        *,
        raw: xr.DataArray,
        steps: Sequence[ToolProvenanceOperation],
        source: tuple[str, xr.DataArray],
        display: tuple[str, ToolProvenanceOperation, xr.DataArray] | None,
    ) -> None:
        """Publish a prepared file source and its displayed value in one change."""
        if not steps:
            self.adopt_checkpoint(
                entry_id, *source, file_load=file_load, backing=backing, display=display
            )
            return
        self._write_generation()
        check_id(entry_id)
        source_key, source_data = source
        check_id(source_key)
        # If the source has already been captured, its record is authoritative.
        # Otherwise stage the whole chain before exposing any of its revision IDs.
        staged: list[_StoredRevision] = []
        stored = self._revisions.get(source_key)
        if stored is None:
            sources = self._sources(file_load)
            raw_key = uuid.uuid4().hex
            owned = checkpoint_data(raw, backing=backing)
            stored = _StoredRevision(
                DataRevision(raw_key, file_load, True, backing),
                raw_key,
                file_load,
                owned,
                (),
            )
            for index, operation in enumerate(steps):
                staged.append(stored)
                key = source_key if index + 1 == len(steps) else uuid.uuid4().hex
                stored = self._record_unary(
                    stored, key, operation, EarlierStep(stored.record.revision_id)
                )
        else:
            sources = self._sources(stored.recipe)
        self._publish_adopted(
            entry_id,
            stored,
            source_data,
            display=display,
            ancestors=staged,
            sources=sources,
        )

    def _publish_adopted(
        self,
        entry_id: str,
        stored: _StoredRevision,
        owned: xr.DataArray,
        *,
        display: tuple[str, ToolProvenanceOperation, xr.DataArray] | None = None,
        auto: bool = False,
        ancestors: Sequence[_StoredRevision] = (),
        sources: set[str] | None = None,
        stale: bool = False,
    ) -> None:
        source_id = stored.record.revision_id
        stored = self._revisions.get(source_id, stored)
        shown, values = stored, owned
        if display is not None:
            key, operation, values = display
            if key == source_id:
                raise ValueError("An accepted filter needs its own revision ID")
            shown = self._record_unary(stored, key, operation, EarlierStep(source_id))
        revision_id = shown.record.revision_id
        current = self._entries.get(entry_id)
        recipe = stored.recipe
        if sources is None:
            sources = self._sources(recipe)
        if current is None:
            self._check_new_entry(entry_id, None)
            entry = ImageToolEntry(
                entry_id,
                revision_id,
                recipe,
                auto,
                None,
                len(self._entries),
                source_id,
            )
        else:
            entry = replace(
                current,
                revision_id=revision_id,
                source_revision_id=source_id,
                recipe=recipe,
                auto=auto,
            )
        self._index_sources(
            entry_id, None if current is None else current.recipe, sources
        )
        for ancestor in ancestors:
            self._revisions[ancestor.record.revision_id] = ancestor
        self._revisions[source_id] = stored
        self._revisions[revision_id] = shown
        self._bind_entry(entry, owned, values)
        if stale:
            self._stale.add(entry_id)
        else:
            self._stale.discard(entry_id)
        self._auto_pending.pop(entry_id, None)
        self._auto_errors.pop(entry_id, None)
        self._generation += 1
        candidates = self._invalidate({entry_id})
        if stale and auto:
            candidates.add(entry_id)
        self._queue_auto(candidates)

    def _hold_visible(self, key: str, data: xr.DataArray) -> _VisibleRevision:
        visible = self._displayed.get(key)
        if visible is None:
            visible = _VisibleRevision(self._revisions[key], data)
            self._displayed[key] = visible
        return visible

    def _bind_entry(
        self,
        entry: ImageToolEntry,
        source_data: xr.DataArray,
        displayed_data: xr.DataArray,
    ) -> None:
        """Hold each window's values until the last window that displays them closes."""
        source = self._hold_visible(entry.source_revision_id, source_data)
        displayed = (
            source
            if entry.source_revision_id == entry.revision_id
            else self._hold_visible(entry.revision_id, displayed_data)
        )
        roots: tuple[_VisibleRevision | _StoredRevision, ...] = (source, displayed)
        if entry.recipe is not None and entry.recipe is not source.stored.recipe:
            # A pending edit may refer to inputs outside the displayed history.
            roots += tuple(
                self._revisions[key] for key in self._recipe_revisions(entry.recipe)
            )
        self._entries[entry.entry_id] = entry
        self._entry_roots[entry.entry_id] = roots

    def _previous_step(
        self, revision_id: str, recipe: Recipe | FileLoad | None
    ) -> Pin | EarlierStep:
        if recipe is None:
            return Pin(revision_id)
        if recipe is self._revisions[revision_id].recipe:
            return EarlierStep(revision_id)
        return EarlierStep(revision_id, PendingRecipe(recipe))

    def _previous_input(
        self, entry: ImageToolEntry, role: ScriptInputDataRole
    ) -> Pin | EarlierStep:
        """Bind a past value while preserving what its next refresh will use."""
        source = self._previous_step(entry.source_revision_id, entry.recipe)
        if role == "displayed" and (view := self._display_recipe(entry)) is not None:
            return self._previous_step(
                entry.revision_id, view._with_inputs((("data", source),))
            )
        return source

    def _display_recipe(self, entry: ImageToolEntry) -> Recipe | None:
        if entry.revision_id == entry.source_revision_id:
            return None
        return typing.cast("Recipe", self._revisions[entry.revision_id].recipe)

    def set_display_operation(
        self,
        entry_id: str,
        operation: ToolProvenanceOperation | None,
        *,
        revision_id: str | None = None,
        data: xr.DataArray | None = None,
    ) -> str:
        """Publish an accepted filter on the source; clearing it restores the source."""
        generation = self._write_generation()
        entry = self._entries[entry_id]
        base_id = entry.source_revision_id
        base = self._revisions[base_id]
        if operation is None:
            if entry.revision_id == base_id:
                return base_id
            shown = base
            values = self._displayed[base_id].data
        else:
            key = revision_id or uuid.uuid4().hex
            check_id(key)
            if key in self._revisions:
                raise ValueError("An accepted filter needs its own revision ID")
            shown = self._record_unary(
                base, key, operation, self._previous_step(base_id, entry.recipe)
            )
            if data is None:
                application = typing.cast("Application", shown.record.producer)
                data = application.apply({base_id: self._displayed[base_id].data})
            values = protected_data(data)
        if self._generation != generation and self._entries.get(entry_id) is not entry:
            raise _SupersededEvaluation("The filter source changed during evaluation")
        key = shown.record.revision_id
        self._revisions[key] = shown
        self._bind_entry(
            replace(entry, revision_id=key), self._displayed[base_id].data, values
        )
        self._generation += 1
        self._queue_auto(self._invalidate({entry_id}))
        return key

    def remove_entry(self, entry_id: str) -> None:
        """Close a window. Followers retain its last displayed revision."""
        self._write_generation()
        entry = self._entries[entry_id]
        self._index_sources(entry_id, entry.recipe, set())
        if self._followers.get(entry_id):
            self._closed[entry_id] = self._revisions[entry.revision_id]
        del self._entries[entry_id], self._entry_roots[entry_id]
        self._stale.discard(entry_id)
        self._auto_pending.pop(entry_id, None)
        self._auto_errors.pop(entry_id, None)
        self._generation += 1

    def file_source(self, path: str) -> FileSource:
        self._write_generation()
        if not path:
            raise ValueError("A file source needs a path")
        source = self._files.get(path)
        if source is None:
            source = FileSource(uuid.uuid4().hex, path)
            self._files[path] = source
        return source

    def derive(
        self,
        entry_id: str,
        recipe: Recipe | FileLoad,
        *,
        auto: bool = False,
        parent_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> str:
        self._check_new_entry(entry_id, parent_id)
        return self._run(
            entry_id,
            copy.deepcopy(recipe),
            auto=auto,
            parent_id=parent_id,
            is_cancelled=is_cancelled,
        )

    def replace(
        self,
        entry_id: str,
        recipe: Recipe | FileLoad,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> str:
        current = self._entries[entry_id]
        recipe = copy.deepcopy(recipe)
        if isinstance(recipe, Recipe):
            # Reading the target consumes its displayed value. On the next refresh,
            # replay its previous producer, or pin an irreplaceable source.
            prior = self._previous_input(current, "displayed")
            recipe = recipe._with_inputs(
                tuple(
                    (slot, prior if binding == Follow(entry_id) else binding)
                    for slot, binding in recipe._captured.inputs
                ),
            )
        return self._run(entry_id, recipe, is_cancelled=is_cancelled)

    def edit_application(
        self,
        entry_id: str,
        revision_id: str,
        operation: ToolProvenanceOperation,
        *,
        changed_inputs: dict[str, Follow] | None = None,
    ) -> str:
        """Edit the displayed calculation and keep the existing Refresh bindings."""
        entry = self._entries[entry_id]
        if entry.source_revision_id != revision_id:
            raise ValueError("The displayed revision changed. Reopen the editor.")
        producer = self._revisions[revision_id].record.producer
        if not isinstance(entry.recipe, Recipe) or not isinstance(
            producer, Application
        ):
            raise TypeError("The entry has no editable operation.")
        changed = {} if changed_inputs is None else changed_inputs
        previous = dict(entry.recipe.inputs)
        used = dict(producer.inputs)
        slots = ("data", *operation.input_slots())
        pending = Recipe(
            operation,
            tuple(
                (slot, changed[slot] if slot in changed else previous[slot])
                for slot in slots
            ),
            execution=entry.recipe._captured.execution,
        )
        displayed = Recipe(
            operation,
            tuple(
                (slot, changed[slot] if slot in changed else Pin(used[slot]))
                for slot in slots
            ),
            execution=producer.execution,
        )
        return self._run(entry_id, pending, displayed=displayed)

    def set_auto(self, entry_id: str, *, enabled: bool) -> None:
        self._write_generation()
        entry = self._entries[entry_id]
        if entry.auto != enabled:
            self._entries[entry_id] = replace(entry, auto=enabled)
            self._generation += 1
        self._auto_errors.pop(entry_id, None)
        if enabled and entry_id in self._stale:
            self._queue_auto([entry_id])
        else:
            self._auto_pending.pop(entry_id, None)

    def set_recipe(self, entry_id: str, recipe: Recipe | FileLoad) -> None:
        """Change the next calculation without changing the displayed revision."""
        self._write_generation()
        current = self._entries[entry_id]
        recipe = copy.deepcopy(recipe)
        sources = self._check_recipe(entry_id, recipe)
        self._index_sources(entry_id, current.recipe, sources)
        self._bind_entry(
            replace(current, recipe=recipe),
            self._displayed[current.source_revision_id].data,
            self._displayed[current.revision_id].data,
        )
        self._auto_errors.pop(entry_id, None)
        self._stale.add(entry_id)
        candidates = self._invalidate({entry_id})
        self._generation += 1
        self._queue_auto([entry_id, *candidates])

    def plan_update(
        self, entry_id: str, *, command: typing.Literal["refresh", "reload"]
    ) -> UpdatePlan:
        """Preview the same request that Refresh or Reload would execute."""
        if command not in {"refresh", "reload"}:
            raise ValueError("Expected refresh or reload")
        entry = self._entries[entry_id]
        return self._plan(Follow(entry_id), recipe=entry.recipe, mode=command).describe(
            self
        )

    def refresh(
        self, entry_id: str, *, is_cancelled: Callable[[], bool] | None = None
    ) -> str:
        """Use available inputs and bring stale prerequisites up to date."""
        return self._run(
            entry_id,
            self._entries[entry_id].recipe,
            mode="refresh",
            is_cancelled=is_cancelled,
        )

    def reload(
        self,
        entry_id: str,
        *,
        is_cancelled: Callable[[], bool] | None = None,
        choose_destination: Callable[[xr.DataArray], str | None] | None = None,
    ) -> str:
        """Reload once and let the host choose a window for the evaluated result.

        The host borrows the result for inspection. Return the current entry ID,
        a new entry ID, or None to cancel the complete update.
        """
        return self._run(
            entry_id,
            self._entries[entry_id].recipe,
            mode="reload",
            is_cancelled=is_cancelled,
            choose_destination=choose_destination,
        )

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
                for _, binding in step._captured.inputs:
                    if isinstance(binding, Follow):
                        if (
                            binding.entry_id not in self._entries
                            and binding.entry_id not in self._closed
                        ):
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
                for _, binding in step._captured.inputs:
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
            for _, binding in recipe._captured.inputs
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
                stored.recipe._captured.inputs, applied.inputs, strict=True
            ):
                if (
                    isinstance(source, Follow)
                    and (
                        self._entries[source.entry_id].revision_id
                        if source.entry_id in self._entries
                        else self._closed[source.entry_id].record.revision_id
                    )
                    != used
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

    def _invalidate(self, changed: set[str]) -> set[str]:
        """Mark followers stale and collect Auto candidates along the affected paths."""
        pending = list(changed)
        candidates: set[str] = set()
        while pending:
            for child in self._followers.get(pending.pop(), ()):
                if child in changed:
                    continue
                if self._entries[child].auto:
                    candidates.add(child)
                # The descendants of an already stale entry are already stale.
                if child not in self._stale:
                    self._stale.add(child)
                    pending.append(child)
        return candidates

    def auto_error(self, entry_id: str) -> str | None:
        """Return the last automatic refresh error for this entry."""
        if entry_id not in self._entries:
            raise KeyError(entry_id)
        return self._auto_errors.get(entry_id)

    def auto_blockers(self, entry_id: str) -> frozenset[str]:
        """Return stale sources that Auto must wait for before refreshing this entry."""
        return frozenset(self._sources(self._entries[entry_id].recipe) & self._stale)

    def _queue_auto(self, candidates: Iterable[str]) -> None:
        for entry_id in candidates:
            if self._entries[entry_id].auto and entry_id in self._stale:
                self._auto_pending[entry_id] = None
        self._schedule_pending_auto()

    def _schedule_pending_auto(self) -> None:
        if self._auto_scheduled or self._posting_auto or not self._auto_pending:
            return
        # An inline scheduler finishes the pass before returning. Post its
        # continuation iteratively; an event-loop scheduler posts one callback.
        self._posting_auto = True
        try:
            while self._auto_pending and not self._auto_scheduled:
                self._auto_scheduled = True
                if self._schedule_auto is None:
                    self._run_auto()
                else:
                    try:
                        self._schedule_auto(self._run_auto)
                    except Exception as exc:
                        for entry_id in self._auto_pending:
                            self._auto_errors[entry_id] = (
                                f"Could not schedule Auto: {exc}"
                            )
                        self._auto_pending.clear()
                        self._auto_scheduled = False
        finally:
            self._posting_auto = False

    def _run_auto(self) -> None:
        self._write_generation()
        attempted: set[str] = set()
        next_pass: dict[str, None] = {}
        try:
            while self._auto_pending:
                entry_id, _ = self._auto_pending.popitem()
                entry = self._entries[entry_id]
                if not entry.auto or entry_id not in self._stale:
                    continue
                # Only an explicit command may move a stale Auto-off prerequisite.
                # Each prerequisite that publishes will queue its followers again.
                if self.auto_blockers(entry_id):
                    continue
                if entry_id in attempted:
                    next_pass[entry_id] = None
                    continue
                attempted.add(entry_id)
                try:
                    self.refresh(entry_id)
                except _SupersededEvaluation:
                    if entry_id in self._stale and self._entries[entry_id].auto:
                        next_pass[entry_id] = None
                except Exception as exc:
                    # This failure belongs to the current inputs. A later input
                    # change or an explicit retry will queue the entry again.
                    self._auto_pending.pop(entry_id, None)
                    if entry_id in self._stale and self._entries[entry_id].auto:
                        self._auto_errors[entry_id] = str(exc) or type(exc).__name__
        finally:
            self._auto_scheduled = False
        self._queue_auto(next_pass)

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
        previous = self._sources(old)
        for source in previous - sources:
            followers = self._followers[source]
            followers.remove(entry_id)
            if not followers:
                del self._followers[source]
                self._closed.pop(source, None)
        for source in sources - previous:
            self._followers.setdefault(source, set()).add(entry_id)

    def _check_current(self, plan: _Plan) -> None:
        for target, (observed, stale) in plan.observed.items():
            if (
                self._entries.get(target) is not observed
                or (target in self._stale) != stale
            ):
                raise _SupersededEvaluation("Provenance changed during evaluation")

    def _write_generation(self) -> int:
        if threading.get_ident() != self._owner:
            raise RuntimeError("Only the graph's owner thread can publish changes")
        return self._generation

    def _run(
        self,
        entry_id: str,
        recipe: Recipe | FileLoad | None,
        *,
        mode: _Mode = "displayed",
        displayed: Recipe | None = None,
        auto: bool = False,
        parent_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        choose_destination: Callable[[xr.DataArray], str | None] | None = None,
    ) -> str:
        generation = self._write_generation()
        if recipe is None and mode != "displayed":
            _check_cancelled(is_cancelled)
            return self._entries[entry_id].revision_id
        sources = (
            self._check_recipe(entry_id, recipe)
            if mode == "displayed"
            else self._sources(recipe)
        )
        root = Follow(entry_id)
        plan = self._plan(
            root,
            recipe=displayed if displayed is not None else recipe,
            mode=mode,
            keep_display=displayed is not None,
        )
        try:
            result = self._evaluate(plan, is_cancelled=is_cancelled)
        except _SupersededEvaluation:
            raise
        except Exception:
            # An operation can fail on an input that changed while it ran.
            if self._generation != generation:
                self._check_current(plan)
            raise
        _check_cancelled(is_cancelled)
        if self._generation != generation:
            self._check_current(plan)
            if mode == "displayed":
                # A new edge elsewhere may have closed a proposed live cycle.
                self._check_recipe(entry_id, recipe)

        forked = False
        if choose_destination is not None:
            destination = choose_destination(result.entries[entry_id].displayed[1])
            if destination is None:
                raise CancelledError("The reloaded data was not published")
            if self._generation != generation:
                self._check_current(plan)
            if destination != entry_id:
                self._check_new_entry(destination, None)
                forked = True
                created = result.entries.pop(entry_id)
                result.entries[destination] = created
                entry_id = destination
                for revision_id in {created.source[0], created.displayed[0]}:
                    result.revisions[revision_id].producer_id = revision_id

        historical = False
        if displayed is not None and isinstance(recipe, Recipe):
            revision_id = result.entries[entry_id].source[0]
            captured = result.revisions[revision_id]
            applied = typing.cast("Application", captured.record.producer)
            parents = {
                revision.record.revision_id: revision for revision in captured.parents
            }
            for key in self._recipe_revisions(recipe):
                parents[key] = self._revisions[key]
            captured.recipe = recipe
            captured.parents = tuple(parents.values())
            historical = any(
                isinstance(binding, Follow)
                and binding.entry_id in self._entries
                and self._entries[binding.entry_id].revision_id != used
                for (_, binding), (_, used) in zip(
                    recipe.inputs, applied.inputs, strict=True
                )
            )
        if self._install is not None:
            updates = tuple(
                value.for_host(target, result.revisions)
                for target, value in result.entries.items()
            )
            installation = self._install(updates)
        else:
            installation = nullcontext()
        with installation:
            _check_cancelled(is_cancelled)
            if self._generation != generation:
                self._check_current(plan)
                if mode == "displayed":
                    self._check_recipe(entry_id, recipe)
                if forked:
                    self._check_new_entry(entry_id, None)

            # Install the values and edges after all operations succeed.
            self._revisions.update(result.revisions)
            for target, value in result.entries.items():
                revision_id, data = value.displayed
                source_id, source_data = value.source
                old = self._entries.get(target)
                effective = result.revisions[source_id].recipe
                if old is None:
                    entry = ImageToolEntry(
                        target,
                        revision_id,
                        effective,
                        auto,
                        parent_id,
                        len(self._entries),
                        source_id,
                    )
                else:
                    entry = replace(
                        old,
                        revision_id=revision_id,
                        source_revision_id=source_id,
                        recipe=effective,
                    )
                if target == entry_id and (mode == "displayed" or forked):
                    self._index_sources(target, old.recipe if old else None, sources)
                self._bind_entry(entry, source_data, data)
                self._auto_errors.pop(target, None)

            candidates = self._invalidate(set(result.entries))
            self._stale.difference_update(result.entries)
            if (
                mode == "displayed"
                and recipe is not None
                and (
                    historical
                    or sources & self._stale
                    or self._earlier_input_changed(recipe)
                )
            ):
                # Derive and Replace consume the data on screen, including stale data.
                self._stale.add(entry_id)
                candidates.add(entry_id)
            self._generation += 1
        self._queue_auto(candidates)
        return self._entries[entry_id].revision_id

    def _plan_input(self, binding: Binding, *, mode: _Mode) -> _Action:
        if isinstance(binding, Follow):
            if binding.entry_id in self._closed:
                return Pin(self._closed[binding.entry_id].record.revision_id)
            entry = self._entries[binding.entry_id]
            if entry.recipe is not None and (
                mode == "reload"
                or (mode == "refresh" and binding.entry_id in self._stale)
            ):
                return entry.recipe
            return Pin(entry.revision_id)
        if isinstance(binding, EarlierStep):
            if mode != "displayed":
                return self._earlier_recipe(binding)
            return Pin(binding.revision_id)

        stored = self._revisions[binding.revision_id]
        visible = self._displayed.get(binding.revision_id)
        data = visible.data if visible is not None else stored.checkpoint
        if data is not None:
            return binding.revision_id, data
        producer = stored.record.producer
        if not isinstance(producer, Application):
            raise TypeError(f"Revision {binding.revision_id!r} needs a checkpoint")
        recipe = typing.cast("Recipe", stored.recipe)
        return recipe._with_inputs(
            tuple((slot, Pin(node)) for slot, node in producer.inputs),
        )

    def _plan(
        self,
        root: Binding,
        *,
        recipe: Recipe | FileLoad | None = None,
        mode: _Mode = "displayed",
        keep_display: bool = False,
    ) -> _Plan:
        plan: dict[Binding, _Action] = {}
        producers: dict[str, list[Follow | EarlierStep]] = {}
        producer_ids: dict[Binding, str] = {}
        dependencies: dict[Binding, tuple[Binding, ...]] = {}
        sources: dict[str, _StoredRevision] = {}
        observed: dict[str, tuple[ImageToolEntry | None, bool]] = {}
        if recipe is not None:
            # Pending edits can pin inputs that Replace does not evaluate.
            sources.update(
                (node, self._revisions[node]) for node in self._recipe_revisions(recipe)
            )
        display_sources: dict[str, EarlierStep] = {}
        requested: dict[Binding, Recipe | FileLoad] = (
            {root: recipe} if recipe is not None else {}
        )
        pending = [root]
        while pending:
            binding = pending.pop()
            if binding in plan:
                continue
            if isinstance(binding, Follow):
                observed[binding.entry_id] = (
                    self._entries.get(binding.entry_id),
                    binding.entry_id in self._stale,
                )
            else:
                sources[binding.revision_id] = self._revisions[binding.revision_id]
            step = (
                requested[binding]
                if binding in requested
                else self._plan_input(binding, mode=mode)
            )
            if (
                isinstance(binding, Follow)
                and (mode != "displayed" or (keep_display and binding == root))
                and isinstance(step, Recipe | FileLoad)
                and (entry := self._entries.get(binding.entry_id)) is not None
                and (view := self._display_recipe(entry)) is not None
            ):
                # Plan the source and accepted filter as two ordinary applications.
                source = EarlierStep(entry.source_revision_id, PendingRecipe(step))
                requested[source] = step
                display_sources[entry.entry_id] = source
                step = view._with_inputs((("data", source),))
            if (
                mode != "displayed"
                and not isinstance(binding, Pin)
                and isinstance(step, Recipe | FileLoad)
            ):
                revision_id = (
                    self._entries[binding.entry_id].revision_id
                    if isinstance(binding, Follow)
                    else binding.revision_id
                )
                producer_id = self._revisions[revision_id].producer_id
                producer_ids[binding] = producer_id
                producers.setdefault(producer_id, []).append(binding)
            plan[binding] = step
            inputs: tuple[Binding, ...] = ()
            if isinstance(step, Binding):
                inputs = (step,)
            elif isinstance(step, Recipe):
                inputs = tuple(source for _, source in step._captured.inputs)
            dependencies[binding] = inputs
            pending.extend(inputs)

        for bindings in producers.values():
            _share_producer(bindings, plan, dependencies)
        order = TopologicalSorter(dependencies)
        return _Plan(
            root,
            tuple((key, plan[key]) for key in order.static_order()),
            sources,
            observed,
            producer_ids,
            display_sources,
        )

    def _evaluate(
        self, plan: _Plan, *, is_cancelled: Callable[[], bool] | None = None
    ) -> _Evaluation:
        result = _Evaluation(sources=plan.sources)
        generation = self._generation
        for binding, step in plan.steps:
            _check_cancelled(is_cancelled)
            if self._generation != generation:
                self._check_current(plan)
                generation = self._generation
            if isinstance(step, tuple):
                result.values[binding] = step
            elif isinstance(step, Binding):
                result.values[binding] = result.values[step]
            else:
                try:
                    self._evaluate_step(
                        result, binding, step, plan.producer_ids.get(binding)
                    )
                except FileNotFoundError as exc:
                    if not isinstance(step, FileLoad):
                        raise
                    displayed = (
                        self._entries.get(plan.root.entry_id)
                        if isinstance(plan.root, Follow)
                        else None
                    )
                    raise MissingFileSource(step.source, displayed, exc) from exc
            if isinstance(binding, Follow) and not isinstance(step, Pin):
                # A live entry publishes both roles even when a producer was shared.
                shown = result.values[binding]
                raw = shown
                if (source := plan.display_sources.get(binding.entry_id)) is not None:
                    raw = result.values[source]
                    # Point the accepted filter at the new source, so the superseded
                    # source chain can be released after installation.
                    stored = result.revisions[shown[0]]
                    filter_recipe = typing.cast("Recipe", stored.recipe)
                    stored.recipe = filter_recipe._with_inputs(
                        (("data", EarlierStep(raw[0])),)
                    )
                    stored.parents = (result.revisions[raw[0]],)
                result.entries[binding.entry_id] = _EntryResult(raw, shown)
        return result

    def _evaluate_step(
        self,
        result: _Evaluation,
        binding: Binding,
        step: Recipe | FileLoad,
        producer_id: str | None,
    ) -> None:
        producer: Application | FileLoad
        backing = None
        if isinstance(step, FileLoad):
            if self._read_file is None:
                raise ValueError(f"No file reader for {step.source.path!r}")
            loaded = self._read_file(step.source, copy.deepcopy(step._captured))
            if isinstance(loaded, tuple):
                array, backing = loaded
                data = checkpoint_data(array, backing=backing)
            else:
                data = checkpoint_data(loaded)
            producer = step
        else:
            inputs = tuple(
                (slot, result.values[source][0])
                for slot, source in step._captured.inputs
            )
            producer = Application(
                step._captured.operation, inputs, step._captured.execution
            )
            values = dict(result.values[source] for _, source in step._captured.inputs)
            data = producer.apply(values)
            if not isinstance(data, xr.DataArray):
                raise TypeError("An operation must produce a DataArray")
            if not isinstance(binding, Pin):
                # A callback may return its own mutable buffers. Capture the result
                # before handing it to a viewer or evaluating another step.
                data = protected_data(data)
        if isinstance(binding, Pin):
            result.values[binding] = (binding.revision_id, data)
            return
        revision_id = uuid.uuid4().hex
        result.values[binding] = (revision_id, data)
        parent_ids = self._recipe_revisions(step)
        if isinstance(producer, Application):
            parent_ids.update(node for _, node in producer.inputs)
        result.revisions[revision_id] = _StoredRevision(
            DataRevision(revision_id, producer, isinstance(step, FileLoad), backing),
            revision_id if producer_id is None else producer_id,
            step,
            data if isinstance(step, FileLoad) else None,
            tuple(
                result.revisions.get(node) or result.sources[node]
                for node in parent_ids
            ),
        )
