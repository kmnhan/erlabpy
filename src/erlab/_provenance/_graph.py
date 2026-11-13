"""Recorded data dependencies and live input bindings for ImageTool."""

from __future__ import annotations

import copy
import functools
import struct
import threading
import typing
import uuid
import weakref
from collections import ChainMap
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from graphlib import TopologicalSorter

import pydantic
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
    capture_display_data,
    check_id,
    checkpoint_data,
    guard_lazy_data,
    lazy_input_guards,
    lazy_input_problem,
)

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from erlab._provenance._records import _LazyReadGuard
    from erlab.interactive.imagetool._provenance._model import (
        FileReplayCall,
        ToolProvenanceOperation,
    )

    _PrepareView = Callable[
        [str, Pin | EarlierStep, xr.DataArray],
        Sequence[Recipe],
    ]


@dataclass(eq=False)
class _StoredRevision:
    record: DataRevision
    # Re-evaluations of one producing step keep its original identity.
    producer_id: str
    recipe: Recipe | FileLoad | None
    checkpoint: xr.DataArray | None
    # Keep exact inputs and earlier steps for as long as this revision is reachable.
    parents: tuple[_StoredRevision, ...]
    live: bool = False
    depends_on_lazy_input: bool = False
    calculation: bool = False
    retained_value: weakref.ReferenceType[xr.DataArray] | None = None


class CapturedRevision:
    """Keep a consumed revision and its ancestors alive without owning its graph."""

    _stored: _StoredRevision | None = None
    source: Follow | None = None
    _calculated_data: xr.DataArray | None = None

    @property
    def revision_id(self) -> str:
        if self._stored is None:
            raise ValueError("No revision was captured")
        return self._stored.record.revision_id

    @property
    def calculated_data(self) -> xr.DataArray:
        """Read the value retained when this calculation completed."""
        if self._calculated_data is None:
            raise ValueError("Expected a completed calculation")
        return (
            checkpoint_data(self._calculated_data)
            if self._calculated_data.chunks is None
            else self._calculated_data.copy(deep=True)
        )

    def __deepcopy__(self, memo: dict[int, object]) -> CapturedRevision:
        return self


_Mode: typing.TypeAlias = typing.Literal["displayed", "refresh", "reload"]
_Action: typing.TypeAlias = Recipe | FileLoad | Binding | tuple[str, xr.DataArray]


@dataclass(frozen=True)
class _PreparedPublication:
    """An evaluated value and the commit that publishes its producing records."""

    _value: xr.DataArray | None
    commit: Callable[[], str]

    @property
    def value(self) -> xr.DataArray:
        if self._value is None:
            raise ValueError("The publication has no evaluated value")
        return (
            checkpoint_data(self._value)
            if self._value.chunks is None
            else self._value.copy(deep=True)
        )

    def __call__(self) -> str:
        return self.commit()


@dataclass(frozen=True)
class WorkerCalculation:
    """One operation and the graph-owned operands available to its worker."""

    owner: str
    _recipe: Recipe
    _input_revision: str
    _input: xr.DataArray
    _source: _StoredRevision

    _execute: Callable[[ToolProvenanceOperation, xr.DataArray], xr.DataArray]
    _output: xr.DataArray | None = field(default=None, init=False, repr=False)

    @property
    def recipe(self) -> Recipe:
        return copy.deepcopy(self._recipe)

    @property
    def data(self) -> xr.DataArray:
        """Return a separate wrapper for inspecting the queued input."""
        if self._input.chunks is None:
            return checkpoint_data(self._input)
        return self._input.copy(deep=True)

    def run(self) -> xr.DataArray:
        """Execute the routine selected at dispatch and keep its result."""
        if self._output is not None:
            raise RuntimeError("The worker calculation already completed")
        output = self._execute(self.recipe.operation, self._input)
        object.__setattr__(self, "_output", capture_display_data(output, live=True))
        return output


class _SupersededEvaluation(RuntimeError):
    """An observed entry changed before the result could be published."""


def _setting_key(value: typing.Any, *, exclude_group: bool = False) -> Hashable:
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
            _setting_key(getattr(value, name))
            for name in type(value).model_fields
            if not (exclude_group and name == "group")
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
                key = (step._captured.inputs, _setting_key(step._captured.operation))
            representative = versions.setdefault(key, representative)
        for binding in group:
            if representative != binding:
                actions[binding] = representative
                dependencies[binding] = (representative,)


@dataclass(frozen=True)
class DisplayedEdit:
    """Changes to one displayed calculation, identified by its recorded revisions."""

    displayed: str
    append_after: str
    append: tuple[ToolProvenanceOperation, ...] = ()
    replacements: Mapping[str, ToolProvenanceOperation | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class UpdatePlan:
    """Planned reads and calculations, and followers affected by their publication."""

    entries: tuple[str, ...]
    earlier_steps: tuple[EarlierStep, ...]
    file_loads: tuple[FileLoad, ...]
    current_inputs: tuple[tuple[str, str], ...]
    pinned_inputs: tuple[str, ...]
    affected_followers: frozenset[str]
    operations: tuple[Recipe, ...] = field(default=(), repr=False, compare=False)
    input_issue: str | None = None


class MissingFileSource(FileNotFoundError):
    def __init__(
        self,
        source: FileSource,
        displayed: ImageToolEntry | None,
        cause: FileNotFoundError,
    ) -> None:
        self.source = source
        self.displayed_revision_id = displayed.revision_id if displayed else None
        message = (
            f"Could not read file source {source.path!r}: {cause}. "
            "Restore the file or edit its load path"
        )
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

    def describe(
        self, graph: ProvenanceGraph, *, include_operations: bool = False
    ) -> UpdatePlan:
        entries = []
        earlier = []
        files = []
        current = []
        pins: dict[str, None] = {}
        for binding, step in self.steps:
            if isinstance(binding, Pin):
                # Reconstructing a pin only uses recorded data.
                continue
            if (
                isinstance(binding, Follow)
                and binding.source
                and not isinstance(step, Pin)
            ):
                # The displayed row and its earlier step describe this calculation.
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
            operations=tuple(
                copy.deepcopy(step)
                for _, step in self.steps
                if isinstance(step, Recipe)
            )
            if include_operations
            else (),
            earlier_steps=tuple(earlier),
            file_loads=tuple(files),
            current_inputs=tuple(current),
            pinned_inputs=tuple(pins),
            affected_followers=frozenset(graph._descendants(entries)).difference(
                entries
            ),
            input_issue=next(
                (
                    issue
                    for _, step in self.steps
                    if isinstance(step, tuple)
                    if (issue := lazy_input_problem(step[1])) is not None
                ),
                None,
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
    entries: dict[str, tuple[str, xr.DataArray]] = field(default_factory=dict)
    view_inputs: dict[str, xr.DataArray] = field(default_factory=dict)


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
        read_file: Callable[[FileSource, FileReplayCall], xr.DataArray] | None = None,
        schedule_auto: Callable[[Callable[[], None]], None] | None = None,
        placement: Callable[[str], tuple[str | None, int] | None] | None = None,
        validate_update: Callable[[str, xr.DataArray], None] | None = None,
        apply_operation: Callable[
            [Application, Mapping[str, xr.DataArray]], xr.DataArray
        ] = Application.apply,
        execution_scope: Callable[[], AbstractContextManager[Callable[[], None]]]
        | None = None,
        install: Callable[
            [Mapping[str, str], Mapping[str, xr.DataArray]],
            AbstractContextManager[Mapping[str, xr.DataArray]],
        ]
        | None = None,
        installed_step: Callable[[str, str, xr.DataArray], Recipe | None] | None = None,
        published: Callable[[Sequence[str]], None] | None = None,
        snapshot_input: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> None:
        self._read_file = read_file
        self._placement = placement
        self._validate_update = validate_update
        self._apply_operation = apply_operation
        self._execution_scope = execution_scope
        self._install = install
        self._installed_step = installed_step
        self._published = published
        self._snapshot_input = snapshot_input
        self._owner = threading.get_ident()
        self._files: dict[str, FileSource] = {}
        self._entries: dict[str, ImageToolEntry] = {}
        self._revisions: weakref.WeakValueDictionary[str, _StoredRevision] = (
            weakref.WeakValueDictionary()
        )
        self._input_aliases: weakref.WeakValueDictionary[str, _StoredRevision] = (
            weakref.WeakValueDictionary()
        )
        self._closed: dict[str, tuple[ImageToolEntry, tuple[_StoredRevision, ...]]] = {}
        self._entry_roots: dict[str, tuple[_StoredRevision, ...]] = {}
        self._displayed: dict[str, xr.DataArray] = {}
        self._upstream: dict[str, frozenset[str]] = {}
        self._followers: dict[str, set[str]] = {}
        self._stale: set[str] = set()
        self._generation = 0
        self._schedule_auto = schedule_auto
        self._auto_pending: dict[str, None] = {}
        self._auto_paused = False
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

    def has_revision(self, revision_id: str) -> bool:
        return revision_id in self._revisions

    def follows(self, entry_id: str) -> frozenset[str]:
        return self._upstream.get(entry_id, frozenset())

    def followers(self, entry_id: str) -> frozenset[str]:
        return frozenset(self._followers.get(entry_id, ()))

    def has_recipe(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        return entry is not None and entry.recipe is not None

    def current_revision(self, entry_id: str) -> str:
        return self._entries[entry_id].revision_id

    def auto_enabled(self, entry_id: str) -> bool:
        return self._entries[entry_id].auto

    def entry(self, entry_id: str) -> ImageToolEntry:
        # A frozen typed operation can still contain mutable parameter values.
        entry = copy.deepcopy(self._entries[entry_id])
        if (
            self._placement is not None
            and (placement := self._placement(entry_id)) is not None
        ):
            return replace(entry, parent_id=placement[0], order=placement[1])
        return entry

    def is_closed(self, uid: str) -> bool:
        return uid in self._closed

    def owner(self, revision_id: str) -> str:
        return self._revisions[revision_id].record.owner_id

    def is_calculation(self, revision_id: str) -> bool:
        """Return whether this result was captured before publication in a window."""
        return self._revisions[revision_id].calculation

    def recipe_for(self, revision_id: str) -> Recipe | FileLoad | None:
        return copy.deepcopy(self._revisions[revision_id].recipe)

    def revision(self, revision_id: str) -> DataRevision:
        return copy.deepcopy(self._revisions[revision_id].record)

    def capture_input(self, binding: Follow, alias: str, holder: object) -> None:
        """Retain the consumed revision for the lifetime of a pending calculation."""
        key = self.followed_revision(binding)
        stored = self._revisions[key]
        self._protect((stored,))
        self._input_aliases[alias] = stored
        # Retain the record alone. Holding the graph here could keep its Manager
        # and every console input in that Manager alive through the finalizer.
        if isinstance(holder, CapturedRevision):
            holder._stored = stored
            holder.source = binding
        else:
            retained = [stored]
            weakref.finalize(holder, retained.clear).atexit = False

    def calculation_input(
        self, revision: str, *, target: str | None = None
    ) -> EarlierStep:
        """Bind a retained calculation, preserving its recorded edges on replacement."""
        root = EarlierStep(revision)
        # A retained expression can outlive its source window. When it replaces
        # one of its inputs, only this consumer must stop following that entry.
        updates: dict[EarlierStep, EarlierStep] = {}
        order: TopologicalSorter[EarlierStep] = TopologicalSorter()
        pending = [root]
        seen = set()
        while pending:
            binding = pending.pop()
            if binding in seen:
                continue
            seen.add(binding)
            step = self._earlier_recipe(binding, self._revisions)
            children = (
                [b for _, b in step._captured.inputs if isinstance(b, EarlierStep)]
                if isinstance(step, Recipe)
                else []
            )
            order.add(binding, *children)
            pending.extend(children)
        for binding in order.static_order():
            step = self._earlier_recipe(binding, self._revisions)
            if not isinstance(step, Recipe):
                continue
            applied = self._revisions[binding.revision_id].record.producer
            if not isinstance(applied, Application):
                # A restored stale checkpoint has a recipe only for its next update.
                continue
            inputs: list[tuple[str, Binding]] = []
            for (slot, source), (_, used) in zip(
                step._captured.inputs, applied.inputs, strict=True
            ):
                if isinstance(source, EarlierStep):
                    source = updates.get(source, source)
                elif isinstance(source, Follow):
                    if source.entry_id not in self._entries:
                        source = Pin(used)
                    elif source.entry_id == target:
                        source = (
                            self._unfiltered_input(self._entries[target])
                            if source.source
                            else self._prior_input(self._entries[target])
                        )
                inputs.append((slot, source))
            if tuple(inputs) != step._captured.inputs:
                updates[binding] = EarlierStep(
                    binding.revision_id, PendingRecipe(step._with_inputs(tuple(inputs)))
                )
        return updates.get(root, root)

    def captured_input(self, alias: str | None) -> str | None:
        stored = self._input_aliases.get(alias) if alias is not None else None
        return None if stored is None else stored.record.revision_id

    @contextmanager
    def retain(self, revision_id: str) -> Iterator[None]:
        """Keep a selected revision and its inputs available across entry updates."""
        saved = self._revisions[revision_id]
        self._protect((saved,))
        try:
            yield
        finally:
            del saved

    def _protect(
        self,
        revisions: Iterable[_StoredRevision],
        *,
        require_current: Callable[[], None] | None = None,
    ) -> None:
        """Seal owned inputs when an operation, pin, or history needs them.

        Eager inputs were isolated at acceptance. Numeric live inputs have
        immutable backing, which the checkpoint callback can share. A worker
        must never receive a borrowed producer allocation.
        """
        pending = list(revisions)
        while pending:
            stored = pending.pop()
            if not stored.live:
                continue
            key = stored.record.revision_id
            data = self._displayed.get(key, stored.checkpoint)
            if data is not None and data.chunks is None:
                generation = self._generation
                owned = (
                    self._snapshot_input(data)
                    if require_current is not None and self._snapshot_input is not None
                    else checkpoint_data(data)
                )
                if self._generation != generation:
                    if require_current is None:
                        raise _SupersededEvaluation(
                            "Provenance changed while capturing an input"
                        )
                    require_current()
                if stored.checkpoint is not None:
                    stored.checkpoint = owned
                if key in self._displayed:
                    self._displayed[key] = owned
            stored.live = False
            pending.extend(stored.parents)

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
                source = self._entries.get(binding.entry_id)
                inputs.append(
                    (
                        slot,
                        binding
                        if (source is None and binding.entry_id not in self._closed)
                        or binding.entry_id in self._stale
                        else self.followed_revision(binding),
                    )
                )
            elif isinstance(binding, Pin):
                inputs.append((slot, binding.revision_id))
            else:
                inputs.append((slot, copy.deepcopy(binding)))
        return tuple(inputs)

    def followed_revision(self, binding: Follow) -> str:
        """Return the exact revision currently supplied by this live binding."""
        entry = self._entries.get(binding.entry_id)
        if entry is None:
            entry = self._closed[binding.entry_id][0]
        if binding.source and entry.source_revision is not None:
            return entry.source_revision
        return entry.revision_id

    def data(self, revision_id: str) -> xr.DataArray:
        """Read a revision, replaying virtual steps from their recorded inputs."""
        root = Pin(revision_id)
        result = self._evaluate(self._plan(root))
        return result.values[root][1].copy(deep=True)

    def calculation_data(self, revision_id: str) -> xr.DataArray:
        """Read a captured operand without copying immutable numeric values."""
        value = self._displayed.get(
            revision_id, self._revisions[revision_id].checkpoint
        )
        if (
            value is None
            and (retained := self._revisions[revision_id].retained_value) is not None
        ):
            value = retained()
        if value is None:
            return self.data(revision_id)
        if value.chunks is not None:
            return value.copy(deep=True)
        return checkpoint_data(value)

    def recorded_lineage(
        self, revision_id: str, *, stop_at: typing.Collection[str] = ()
    ) -> tuple[DataRevision, ...]:
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
                if node not in stop_at and isinstance(producer, Application)
                else ()
            )
            order.add(node, *inputs)
            pending.extend(reversed(inputs))
        return tuple(self.revision(node) for node in order.static_order())

    def checkpoint_problem(self, revision_id: str) -> str | None:
        data = self._revisions[revision_id].checkpoint
        return (
            lazy_input_problem(data)
            if data is not None and data.chunks is not None
            else None
        )

    def checkpoint_is_materialized(self, revision_id: str) -> bool:
        """Return whether the checkpoint data and coordinates are already in memory."""
        data = self._revisions[revision_id].checkpoint
        return data is not None and all(
            variable._in_memory
            for variable in (data.variable, *(c.variable for c in data.coords.values()))
        )

    def checkpoint_reader(self, revision_id: str) -> Callable[[], xr.DataArray] | None:
        """Retain an isolated reader of an already sealed checkpoint."""
        stored = self._revisions[revision_id]
        data = self._displayed.get(revision_id, stored.checkpoint)
        if data is None or stored.live:
            return None
        return functools.partial(data.copy, deep=True)

    def displayed_file_guards(self, entry_id: str) -> tuple[_LazyReadGuard, ...]:
        """Return file checks for the displayed computation without evaluating it."""
        return lazy_input_guards(self._displayed[self.current_revision(entry_id)])

    def recorded_code_inputs(self, revision_id: str) -> set[str]:
        """Cut an unavailable history branch at its first exact materialized result."""
        checkpoints = set()
        missing: dict[str, str] = {}
        for record in self.recorded_lineage(revision_id):
            key = record.revision_id
            stored = self._revisions[key]
            if isinstance(record.producer, Application):
                reason = next(
                    (
                        missing[node]
                        for _, node in record.producer.inputs
                        if node in missing
                    ),
                    None,
                )
            else:
                reason = None
            data = stored.checkpoint
            if data is not None and data.chunks is not None:
                reason = self.checkpoint_problem(key) or reason
            if reason is not None:
                if data is not None and data.chunks is None:
                    checkpoints.add(key)
                else:
                    missing[key] = reason
        return checkpoints

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
        auto: bool = False,
        parent_id: str | None = None,
    ) -> str:
        self._check_new_entry(entry_id, parent_id)
        return self._prepare_record(
            entry_id, data, (), auto=auto, parent_id=parent_id
        )()

    def observe_calculation(
        self, entry_id: str, capture: CapturedRevision
    ) -> Callable[[], None]:
        """Guard the destination and followed inputs during window preparation."""
        binding = self.calculation_input(capture.revision_id, target=entry_id)
        sources = self._sources(self._earlier_recipe(binding, self._revisions))
        return self._observe_entries({*sources, entry_id})

    def _observe_entries(self, entries: set[str]) -> Callable[[], None]:
        observed = {uid: self._entries.get(uid) for uid in entries}

        def check() -> None:
            if any(
                self._entries.get(uid) is not before for uid, before in observed.items()
            ):
                raise _SupersededEvaluation("Provenance changed during publication")

        return check

    def prepare_calculation(
        self,
        entry_id: str,
        capture: CapturedRevision,
        *,
        parent_id: str | None,
        output_name: str | None,
        auto: bool,
        prepare_view: _PrepareView | None = None,
    ) -> Callable[[], str]:
        """Display an existing calculation and retain its exact input edges."""
        self._write_generation()
        binding = self.calculation_input(capture.revision_id, target=entry_id)
        stored = self._revisions[binding.revision_id]
        recipe = self._earlier_recipe(binding, self._revisions)
        if not isinstance(recipe, Recipe):
            raise TypeError("Expected a calculation recipe")
        if entry_id not in self:
            self._check_new_entry(entry_id, parent_id)
        sources = self._sources(recipe)
        self._check_cycle(entry_id, sources)
        require_current = self._observe_entries({*sources, entry_id})
        result = _Evaluation(
            revisions={binding.revision_id: stored},
            entries={entry_id: (binding.revision_id, capture.calculated_data)},
            sources={
                key: self._revisions[key] for key in self._recipe_revisions(recipe)
            },
        )

        def commit() -> str:
            self._write_generation()
            require_current()
            self._check_cycle(entry_id, sources)
            return self._publish(
                entry_id,
                result,
                recipe,
                sources,
                "displayed",
                auto,
                parent_id,
                entry_recipe=recipe,
                output_name=output_name,
                prepare_view=prepare_view,
                next_recipe=recipe
                if self._earlier_steps_changed([(binding, binding.revision_id)])
                else None,
            )

        return commit

    def capture_worker(
        self,
        owner: str,
        recipe: Recipe,
        execute: Callable[[ToolProvenanceOperation, xr.DataArray], xr.DataArray],
    ) -> WorkerCalculation:
        """Resolve the routine's exact graph input on the owner thread."""
        self._write_generation()
        ((_, binding),) = recipe.inputs
        key = (
            self.followed_revision(binding)
            if isinstance(binding, Follow)
            else binding.revision_id
        )
        return WorkerCalculation(
            owner,
            copy.deepcopy(recipe),
            key,
            self.calculation_data(key),
            self._revisions[key],
            execute,
        )

    def retain_completed(self, task: WorkerCalculation) -> CapturedRevision:
        """Register the worker's evaluated request without executing it again."""
        if task._output is None:
            raise ValueError("The worker has no completed calculation")
        key = task._input_revision
        ((slot, _),) = task._recipe.inputs
        if self._revisions.get(key) is not task._source:
            raise ValueError("Worker input does not belong to this graph")
        holder = CapturedRevision()
        self._prepare_record(
            task.owner,
            task._output,
            (task.recipe,),
            calculation_holder=holder,
            window_calculation=True,
            recorded_inputs={(0, slot): key},
        )()
        return holder

    def prepare_source(
        self,
        entry_id: str,
        data: xr.DataArray,
        *,
        file: FileLoad | None = None,
        parent_id: str | None = None,
        next_recipe: Recipe | FileLoad | None = None,
        imported_history: str | None = None,
        prepare_view: _PrepareView | None = None,
    ) -> Callable[[], str]:
        """Accept an external array or the selected output of a file reader."""
        return self._prepare_record(
            entry_id,
            data,
            () if file is None else (file,),
            parent_id=parent_id,
            live=True,
            next_recipe=next_recipe,
            imported_history=imported_history,
            prepare_view=prepare_view,
        )

    def prepare_import(
        self,
        entry_id: str,
        data: xr.DataArray | None,
        steps: Sequence[Recipe | FileLoad],
        *,
        auto: bool = False,
        parent_id: str | None = None,
        file_data: xr.DataArray | None = None,
        filtered: bool = False,
        live: bool = False,
        execution_scope: Callable[[], AbstractContextManager[Callable[[], None]]]
        | None = None,
        imported_history: str | None = None,
        checkpoint_histories: Mapping[str, str] | None = None,
        checkpoints: Mapping[str, tuple[str, xr.DataArray]] | None = None,
        recorded_inputs: Mapping[tuple[int, str], str] | None = None,
        next_recipe: Recipe | FileLoad | None = None,
        file_origins: Mapping[str, FileLoad] | None = None,
        intermediates: Mapping[str, tuple[Recipe, Mapping[str, str]]] | None = None,
        prepare_view: _PrepareView | None = None,
    ) -> Callable[[], str]:
        """Read an externally supplied result or apply a transferred workflow."""
        return self._prepare_record(
            entry_id,
            data,
            steps,
            auto=auto,
            parent_id=parent_id,
            file_data=file_data,
            filtered=filtered,
            live=live,
            execution_scope=execution_scope,
            imported_history=imported_history,
            checkpoint_histories=checkpoint_histories,
            checkpoints=checkpoints,
            recorded_inputs=recorded_inputs,
            next_recipe=next_recipe,
            file_origins=file_origins,
            intermediates=intermediates,
            prepare_view=prepare_view,
        )

    def prepare_evaluation(
        self,
        entry_id: str,
        steps: Sequence[Recipe | FileLoad],
        *,
        parent_id: str | None = None,
        auto: bool = False,
        recorded_inputs: Mapping[tuple[int, str], str] | None = None,
        calculation_holder: CapturedRevision | None = None,
        window_calculation: bool = False,
        checkpoints: Mapping[str, tuple[str, xr.DataArray]] | None = None,
        file_data: xr.DataArray | None = None,
        prepare_view: _PrepareView | None = None,
        next_recipe: Recipe | FileLoad | None = None,
        intermediates: Mapping[str, tuple[Recipe, Mapping[str, str]]] | None = None,
    ) -> _PreparedPublication:
        """Evaluate exact input edges now and return their guarded publication."""
        if not steps or not isinstance(steps[-1], Recipe):
            raise ValueError("A live calculation requires a producing operation")
        return self._prepare_record(
            entry_id,
            None,
            steps,
            parent_id=parent_id,
            auto=auto,
            evaluate=True,
            live=True,
            recorded_inputs=recorded_inputs,
            calculation_holder=calculation_holder,
            window_calculation=window_calculation,
            checkpoints=checkpoints,
            file_data=file_data,
            prepare_view=prepare_view,
            next_recipe=next_recipe,
            intermediates=intermediates,
        )

    def _prepare_record(
        self,
        entry_id: str,
        data: xr.DataArray | None,
        steps: Sequence[Recipe | FileLoad],
        *,
        evaluate: bool = False,
        auto: bool = False,
        parent_id: str | None = None,
        file_data: xr.DataArray | None = None,
        filtered: bool = False,
        live: bool = False,
        execution_scope: Callable[[], AbstractContextManager[Callable[[], None]]]
        | None = None,
        calculation_holder: CapturedRevision | None = None,
        window_calculation: bool = False,
        imported_history: str | None = None,
        checkpoint_histories: Mapping[str, str] | None = None,
        checkpoints: Mapping[str, tuple[str, xr.DataArray]] | None = None,
        recorded_inputs: Mapping[tuple[int, str], str] | None = None,
        next_recipe: Recipe | FileLoad | None = None,
        file_origins: Mapping[str, FileLoad] | None = None,
        intermediates: Mapping[str, tuple[Recipe, Mapping[str, str]]] | None = None,
        prepare_view: _PrepareView | None = None,
    ) -> _PreparedPublication:
        """Publish an already calculated result with its ordered producing steps.

        A reference to the target consumes the preceding step, or its old displayed
        value in the first step. Intermediate steps have revisions but no windows.
        A file load at the start of a pipeline needs its exact selected file data.
        Pass no output data to evaluate the recorded operations upon commit.
        An empty pipeline records a new memory origin. Eager input is isolated
        before returning; live inputs use immutable backing for later sharing.
        Optional intermediate revisions precede the output pipeline in dependency
        order. Each has a recipe and any older input revisions used at capture.
        Lazy expressions must read stable backing values while retained.
        """
        if calculation_holder is not None and (
            (data is None and not evaluate)
            or not steps
            or (entry_id in self and not window_calculation)
        ):
            raise ValueError(
                "A retained calculation requires data, steps, and a fresh ID"
            )
        if (
            data is None
            and not evaluate
            and (
                entry_id not in self
                or not steps
                or any(not isinstance(step, Recipe) for step in steps)
            )
        ):
            raise ValueError("An unevaluated pipeline requires recorded operations")
        generation = self._write_generation()
        old = self._entries.get(entry_id)
        if old is None:
            if entry_id in self._closed:
                raise ValueError(
                    f"ImageTool ID {entry_id!r} still identifies a closed input"
                )
            check_id(entry_id)
            if parent_id is not None:
                check_id(parent_id)
        result = _Evaluation()
        revisions = ChainMap(result.revisions, self._revisions)
        for key, (owner, value) in (checkpoints or {}).items():
            check_id(key)
            if key not in revisions and key not in (intermediates or {}):
                file = (file_origins or {}).get(key)
                if (
                    file is not None
                    and self._files.get(file.source.path) != file.source
                ):
                    raise ValueError("Unknown recorded file source")
                result.revisions[key] = _StoredRevision(
                    DataRevision(
                        key,
                        copy.deepcopy(file),
                        True,
                        owner,
                        (checkpoint_histories or {}).get(key),
                    ),
                    key,
                    copy.deepcopy(file),
                    capture_display_data(
                        value, source_path=file.source.path if file else None, live=live
                    ),
                    (),
                    depends_on_lazy_input=value.chunks is not None
                    or not value.variable._in_memory,
                )
        file_path = (
            steps[0].source.path if steps and isinstance(steps[0], FileLoad) else None
        )
        owned = (
            capture_display_data(data, source_path=file_path, live=live)
            if data is not None
            else None
        )
        values: dict[str, xr.DataArray] = {}

        def input_value(key: str) -> xr.DataArray:
            if key in values:
                return values[key]
            stored = result.revisions.get(key)
            if stored is not None and stored.checkpoint is not None:
                checkpoint = stored.checkpoint
                values[key] = (
                    checkpoint_data(checkpoint)
                    if checkpoint.chunks is None
                    else checkpoint.copy(deep=True)
                )
            else:
                values[key] = self.calculation_data(key)
            return values[key]

        previous = None if old is None else self._prior_input(old)
        overrides = dict(recorded_inputs or {})
        recorded_stale = False
        pipeline = tuple(steps)
        if (
            filtered
            and pipeline
            and isinstance(last := pipeline[-1], Recipe)
            and last.view_role is None
        ):
            pipeline = (
                *pipeline[:-1],
                Recipe(
                    last.operation,
                    last.inputs,
                    input_names=last.input_names,
                    output_name=last.output_name,
                    source_kind=last.source_kind,
                    view_role="filter",
                ),
            )
        staged = list((intermediates or {}).items())
        records: list[tuple[str, Recipe | FileLoad | None, dict[str, str]]] = [
            (key, recipe, dict(actual)) for key, (recipe, actual) in staged
        ]
        records.extend((uuid.uuid4().hex, step, {}) for step in pipeline or (None,))
        for index, (revision_id, step, actual) in enumerate(
            copy.deepcopy(records), start=-len(staged)
        ):
            check_id(revision_id)
            if revision_id in revisions:
                raise ValueError("Duplicate data revision")
            for slot, key in actual.items():
                overrides[index, slot] = key
            producer: Application | FileLoad | None = None
            parents: set[str] = set()
            checkpoint = None
            if isinstance(step, Recipe):
                if previous is not None:
                    step = self._bind_prior(step, entry_id, previous)
                current_edges = tuple(
                    (
                        slot,
                        self.followed_revision(binding)
                        if isinstance(binding, Follow)
                        else binding.revision_id,
                    )
                    for slot, binding in step._captured.inputs
                )
                edges = (
                    tuple(
                        (slot, overrides.pop((index, slot), source))
                        for slot, source in current_edges
                    )
                    if overrides
                    else current_edges
                )
                if edges != current_edges:
                    # The caller calculated this result before a followed input
                    # changed. Keep its actual inputs and the live recipe separately.
                    recorded_stale = True
                producer = Application(step._captured.operation, edges)
                if evaluate:
                    operands = {key: input_value(key) for _, key in edges}
                    value = self._apply_operation(producer, operands)
                    if self._generation != generation:
                        raise _SupersededEvaluation(
                            "Provenance changed during the calculation"
                        )
                    if not isinstance(value, xr.DataArray):
                        raise TypeError("An operation must produce a DataArray")
                    values[revision_id] = value
                    if index == len(pipeline) - 1:
                        owned = capture_display_data(value, live=True)
                    if step.view_role is not None:
                        parent_key = edges[0][1]
                        if revisions[parent_key].checkpoint is None:
                            result.view_inputs[parent_key] = capture_display_data(
                                operands[parent_key], live=True
                            )
                parents = self._recipe_revisions(step)
                parents.update(node for _, node in edges)
            elif isinstance(step, FileLoad):
                if index or self._files.get(step.source.path) != step.source:
                    raise ValueError(
                        "Expected a registered file source as the first step"
                    )
                producer = step
                if len(pipeline) == 1:
                    checkpoint = owned
                elif file_data is None:
                    raise ValueError("A file pipeline requires its loaded source data")
                else:
                    checkpoint = capture_display_data(
                        file_data, source_path=step.source.path
                    )
            else:
                checkpoint = owned
            predecessors = tuple(revisions[node] for node in parents)
            lazy_input = any(parent.depends_on_lazy_input for parent in predecessors)
            if (
                lazy_input
                and index == len(pipeline) - 1
                and owned is not None
                and owned.chunks is None
            ):
                # The exact selected result already exists. Preserve it if its
                # deferred source can later become unavailable.
                checkpoint = owned
            owner_id = entry_id
            if index < 0 and checkpoints is not None and revision_id in checkpoints:
                owner_id, captured_value = checkpoints[revision_id]
                checkpoint = capture_display_data(captured_value)
            stored = _StoredRevision(
                DataRevision(
                    revision_id,
                    producer,
                    checkpoint is not None,
                    owner_id,
                    imported_history if producer is None else None,
                ),
                revision_id,
                step,
                checkpoint,
                predecessors,
                live=live,
                calculation=calculation_holder is not None and not window_calculation,
                depends_on_lazy_input=lazy_input
                if checkpoint is None
                else checkpoint.chunks is not None,
            )
            if not live or window_calculation:
                self._protect(stored.parents)
            result.revisions[revision_id] = stored
            if index >= 0:
                previous = EarlierStep(revision_id)

        if overrides:
            raise ValueError("Recorded inputs must identify a published operation slot")
        if filtered and (
            not isinstance(step, Recipe)
            or not isinstance(step._captured.inputs[0][1], EarlierStep | Pin)
        ):
            raise ValueError("A display filter must refer to its unfiltered revision")
        if recorded_stale and next_recipe is None:
            next_recipe = step
        if next_recipe is not None:
            next_recipe = copy.deepcopy(next_recipe)
            result.sources.update(
                (key, revisions[key]) for key in self._recipe_revisions(next_recipe)
            )
        if not live or window_calculation:
            self._protect(result.sources.values())
        sources = self._sources(next_recipe or step, revisions=revisions)
        self._check_cycle(entry_id, sources)
        if self._generation != generation:
            raise _SupersededEvaluation("Provenance changed during publication")
        if owned is not None and owned.chunks is not None:
            # Calculations may discard encoding['source']. The declared inputs
            # still supply the file versions needed by the already computed
            # expression. A materialized checkpoint ends that dependency.
            guards: list[_LazyReadGuard] = []
            pending = list(stored.parents)
            seen = set()
            while pending:
                parent = pending.pop()
                key = parent.record.revision_id
                if key in seen or not parent.depends_on_lazy_input:
                    continue
                seen.add(key)
                value = self._displayed.get(key, parent.checkpoint)
                if value is not None:
                    guards.extend(lazy_input_guards(value))
                if parent.checkpoint is None:
                    pending.extend(parent.parents)
            owned = guard_lazy_data(owned, guards)
        if owned is not None:
            result.entries[entry_id] = revision_id, owned
        observed_sources = (
            sources | self._sources(step, revisions=revisions)
            if next_recipe is not None
            else sources
        )
        observed = {
            uid: self._entries.get(uid) for uid in {*observed_sources, entry_id}
        }

        root = Pin(revision_id)

        def commit() -> str:
            self._write_generation()
            if any(
                self._entries.get(uid) is not entry for uid, entry in observed.items()
            ):
                raise _SupersededEvaluation("Provenance changed during publication")
            if self._generation != generation:
                self._check_cycle(entry_id, sources)
            if calculation_holder is not None:
                # A Python value owns the exact result without creating a window
                # or registering an Auto follower. Its consumers retain it by edge.
                calculation_holder._calculated_data = owned
                if owned is not None:
                    stored.retained_value = weakref.ref(owned)
                if not window_calculation:
                    stored.checkpoint = owned
                    stored.record = replace(stored.record, checkpointed=True)
                    stored.depends_on_lazy_input = (
                        owned is not None and owned.chunks is not None
                    )
                self._revisions.update(result.revisions)
                calculation_holder._stored = stored
                return revision_id
            if owned is None:
                with self._displayed_transaction(
                    execution_scope, next_recipe=next_recipe
                ) as evaluated:
                    evaluated.sources.update(result.sources)
                    # Make the proposed records available to the pin evaluator.
                    # They remain private to this request until installation.
                    evaluated.revisions.update(result.revisions)
                    pinned = self._evaluate(self._plan(root, revisions=revisions))
                    evaluated.sources.update(pinned.sources)
                    evaluated.values.update(pinned.values)
                    evaluated.entries[entry_id] = pinned.values[root]
                return self.current_revision(entry_id)
            return self._publish(
                entry_id,
                result,
                step,
                sources,
                "displayed",
                auto,
                parent_id,
                next_recipe=next_recipe,
                prepare_view=prepare_view,
            )

        return _PreparedPublication(owned, commit)

    @property
    def generation(self) -> int:
        return self._generation

    def replacement_recipe(self, uid: str) -> Recipe | FileLoad | None:
        """Keep the external inputs a manual replacement can update from."""
        recipe = self._entries[uid].recipe
        if recipe is not None and (
            self.follows(uid)
            or isinstance(recipe, FileLoad)
            or self.plan_update(uid, command="reload").file_loads
        ):
            return recipe
        return None

    def prepare_raw_replacement(
        self,
        uid: str,
        raw_data: xr.DataArray,
        operations: Sequence[ToolProvenanceOperation],
        *,
        raw_revision: str,
        preparation: Sequence[ToolProvenanceOperation],
        keep_recipe: bool,
        prepare_view: _PrepareView,
    ) -> _PreparedPublication:
        """Record the accepted input and keep this entry's live update bindings."""
        base = raw_revision
        first: Binding = Pin(raw_revision)
        intermediates = {}
        original_recipe = self.replacement_recipe(uid) if keep_recipe else None
        if not operations:
            if not preparation:
                raise ValueError(
                    "A calculated replacement requires a preparation or assignment"
                )
            return self.prepare_evaluation(
                uid,
                tuple(
                    Recipe.bind(op, first if i == 0 else Follow(uid))
                    for i, op in enumerate(preparation)
                ),
                checkpoints={raw_revision: (uid, raw_data)},
                next_recipe=original_recipe,
                prepare_view=prepare_view,
            )
        for operation in preparation:
            previous = base
            base = uuid.uuid4().hex
            intermediates[base] = Recipe.bind(operation, first), {"data": previous}
            first = EarlierStep(base)
        if original_recipe is not None:
            # A manual replacement can arrive while the window follows a
            # calculation. Preserve the part preceding the retained assignments.
            # If they came from inside that calculation, apply them to its output.
            pending = original_recipe
            for op in reversed(operations):
                if (
                    not isinstance(pending, Recipe)
                    or len(pending.inputs) != 1
                    or _setting_key(pending._captured.operation) != _setting_key(op)
                ):
                    first = EarlierStep(base, PendingRecipe(original_recipe))
                    break
                original = pending._captured.inputs[0][1]
                if not isinstance(original, EarlierStep):
                    first = original
                    break
                pending = self._earlier_recipe(original, self._revisions)
            else:
                first = EarlierStep(base, PendingRecipe(pending))
        steps = tuple(
            Recipe.bind(op, first if index == 0 else Follow(uid))
            for index, op in enumerate(operations)
        )
        return self.prepare_evaluation(
            uid,
            steps,
            checkpoints={raw_revision: (uid, raw_data)},
            recorded_inputs={(0, "data"): base},
            intermediates=intermediates,
            next_recipe=original_recipe,
            prepare_view=prepare_view,
        )

    def view_input(self, entry_id: str, depth: int) -> EarlierStep | Pin:
        """Recover the displayed input and pending calculation below a view."""
        return self._view_input(self._prior_input(self._entries[entry_id]), depth)

    def _view_input(self, prior: EarlierStep | Pin, depth: int) -> EarlierStep | Pin:
        for _ in range(depth):
            producer = self._revisions[prior.revision_id].record.producer
            if not isinstance(prior, EarlierStep) or not isinstance(
                producer, Application
            ):
                raise TypeError("A display operation must have a recorded input")
            recipe = self._earlier_recipe(prior, self._revisions)
            if not isinstance(recipe, Recipe):
                raise TypeError("A display operation must have a data recipe")
            source = recipe._captured.inputs[0][1]
            if not isinstance(source, EarlierStep | Pin):
                raise TypeError("A display operation must use an earlier data state")
            actual = producer.inputs[0][1]
            if actual == source.revision_id:
                prior = source
            else:
                # A successful refresh rebuilt this earlier step. Its new
                # revision already contains the applied edit.
                prior = (
                    EarlierStep(actual)
                    if self._revisions[actual].recipe is not None
                    else Pin(actual)
                )
        return copy.deepcopy(prior)

    def _unfiltered_input(self, entry: ImageToolEntry) -> EarlierStep | Pin:
        """Bind the public source layout to the base of the pending display recipe."""
        prior = self._prior_input(entry)
        key = entry.source_revision
        if key is None:
            return prior
        recipe = entry.recipe
        if isinstance(recipe, Recipe) and recipe.view_role == "filter":
            return self._view_input(prior, 1)
        if isinstance(recipe, Recipe) and recipe.view_role == "layout":
            base = self._view_input(prior, 2)
            layout = typing.cast("Recipe", self._revisions[key].recipe)
            if layout._captured.inputs == (("data", base),) and _setting_key(
                layout._captured.operation
            ) == _setting_key(recipe._captured.operation):
                return EarlierStep(key)
            return EarlierStep(
                key, PendingRecipe(recipe._with_inputs((("data", base),)))
            )
        return self._revision_input(key, recipe)

    def _unfiltered_revision(
        self,
        target: str,
        displayed: _StoredRevision,
        result: _Evaluation,
        old_source: str | None,
    ) -> _StoredRevision | None:
        """Keep the unfiltered array in the same public layout as the display."""
        step = displayed.recipe
        if not isinstance(step, Recipe) or step.view_role is None:
            return None
        producer = typing.cast("Application", displayed.record.producer)
        parent = self._revisions[producer.inputs[0][1]]
        if step.view_role == "filter":
            return parent
        if not isinstance(parent.recipe, Recipe) or parent.recipe.view_role != "filter":
            return None
        filtered = typing.cast("Application", parent.record.producer)
        base = self._revisions[filtered.inputs[0][1]]
        # A source consumer in this Reload may already have evaluated this layout.
        evaluated = result.values.get(Follow(target, source=True))
        for candidate in (evaluated[0] if evaluated else None, old_source):
            if candidate is not None:
                existing = self._revisions[candidate]
                applied = existing.record.producer
                if (
                    isinstance(applied, Application)
                    and applied.inputs == filtered.inputs
                    and _setting_key(applied.operation)
                    == _setting_key(producer.operation)
                ):
                    return existing
        key = uuid.uuid4().hex
        recipe = step._with_inputs(
            (("data", self._revision_input(base.record.revision_id, base.recipe)),)
        )
        source = _StoredRevision(
            DataRevision(
                key, Application(producer.operation, filtered.inputs), False, target
            ),
            key,
            recipe,
            None,
            (base,),
            depends_on_lazy_input=base.depends_on_lazy_input,
        )
        self._revisions[key] = source
        return source

    def restore_view(self, entry_id: str, prior: EarlierStep | Pin) -> None:
        """Remove display operations and restore the original refresh binding."""
        self._write_generation()
        revision_id = prior.revision_id
        stored = self._revisions[revision_id]
        if stored.record.owner_id != entry_id and not stored.calculation:
            raise ValueError("The view belongs to another entry")
        recipe = (
            self._earlier_recipe(prior, self._revisions)
            if isinstance(prior, EarlierStep)
            else None
        )
        stale = isinstance(prior, EarlierStep) and self._earlier_steps_changed(
            [(prior, revision_id)]
        )
        sources = self._sources(recipe)
        self._check_cycle(entry_id, sources)
        value = (
            stored.checkpoint
            if stored.checkpoint is not None
            else self.calculation_data(revision_id)
        )
        result = _Evaluation(
            revisions={revision_id: stored}, entries={entry_id: (revision_id, value)}
        )
        if stale and recipe is not None:
            result.sources.update(
                (key, self._revisions[key]) for key in self._recipe_revisions(recipe)
            )
        self._publish(
            entry_id,
            result,
            recipe,
            sources,
            "displayed",
            False,
            None,
            next_recipe=recipe if stale else None,
        )

    def promote_entry(self, entry_id: str) -> None:
        """Move an entry to the top level while retaining its data and bindings."""
        self._write_generation()
        entry = self._entries[entry_id]
        if entry.parent_id is not None:
            self._entries[entry_id] = replace(entry, parent_id=None)
            self._generation += 1

    def duplicate_entries(
        self,
        owners: Mapping[str, str],
        *,
        origin_histories: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Copy a subtree's history and rebind its internal inputs to the copies."""
        self._write_generation()
        if not owners:
            return {}
        if len(set(owners.values())) != len(owners):
            raise ValueError("Each duplicate needs a distinct entry ID")
        roots: list[_StoredRevision] = []
        for old, new in owners.items():
            self._check_new_entry(new, None)
            roots.extend(self._entry_roots[old])
        self._protect(roots)
        pending = list(roots)
        order: TopologicalSorter[str] = TopologicalSorter()
        keys: dict[str, str] = {}
        while pending:
            stored = pending.pop()
            key = stored.record.revision_id
            if key in keys or stored.record.owner_id not in owners:
                continue
            keys[key] = uuid.uuid4().hex
            order.add(key, *(parent.record.revision_id for parent in stored.parents))
            pending.extend(stored.parents)

        def rebind(recipe: Recipe) -> Recipe:
            inputs: list[tuple[str, Binding]] = []
            for slot, binding in recipe._captured.inputs:
                if isinstance(binding, Follow):
                    binding = replace(
                        binding, entry_id=owners.get(binding.entry_id, binding.entry_id)
                    )
                else:
                    old_key = binding.revision_id
                    binding = replace(binding, revision_id=keys.get(old_key, old_key))
                    if (
                        isinstance(binding, EarlierStep)
                        and binding.pending
                        and isinstance(binding.pending.recipe, Recipe)
                    ):
                        binding = replace(
                            binding,
                            pending=PendingRecipe(rebind(binding.pending.recipe)),
                        )
                inputs.append((slot, binding))
            return recipe._with_inputs(tuple(inputs))

        copies: dict[str, _StoredRevision] = {}
        producers: dict[str, str] = {}
        for key in order.static_order():
            if key not in keys:
                continue
            source = self._revisions[key]
            record = source.record
            applied = record.producer
            if isinstance(applied, Application):
                applied = Application(
                    applied.operation,
                    tuple(
                        (slot, keys.get(source_key, source_key))
                        for slot, source_key in applied.inputs
                    ),
                )
            copies[key] = replace(
                source,
                record=replace(
                    record,
                    revision_id=keys[key],
                    owner_id=owners[record.owner_id],
                    producer=applied,
                    imported_history=(origin_histories or {}).get(
                        key, record.imported_history
                    ),
                ),
                producer_id=producers.setdefault(source.producer_id, keys[key]),
                recipe=rebind(source.recipe)
                if isinstance(source.recipe, Recipe)
                else source.recipe,
                parents=tuple(
                    copies.get(parent.record.revision_id, parent)
                    for parent in source.parents
                ),
            )
        copied_records = {item.record.revision_id: item for item in copies.values()}
        revisions = ChainMap(copied_records, self._revisions)
        entries: list[tuple[str, ImageToolEntry, set[str]]] = []
        for old, new in owners.items():
            original = self._entries[old]
            recipe = (
                rebind(original.recipe)
                if isinstance(original.recipe, Recipe)
                else original.recipe
            )
            entry = replace(
                original,
                entry_id=new,
                revision_id=keys.get(original.revision_id, original.revision_id),
                parent_id=owners.get(original.parent_id, original.parent_id)
                if original.parent_id is not None
                else None,
                order=len(self._entries) + len(entries),
                recipe=recipe,
                source_revision=keys.get(
                    original.source_revision, original.source_revision
                )
                if original.source_revision is not None
                else None,
            )
            entries.append(
                (
                    old,
                    entry,
                    self._sources(recipe, revisions=revisions, allow_missing=True),
                )
            )
        self._revisions.update(copied_records)
        for old, entry, sources in entries:
            self._entries[entry.entry_id] = entry
            self._entry_roots[entry.entry_id] = tuple(
                copies.get(root.record.revision_id, root)
                for root in self._entry_roots[old]
            )
            self._displayed[entry.revision_id] = self._displayed[
                self._entries[old].revision_id
            ]
            self._index_sources(entry.entry_id, sources)
            if old in self._stale:
                self._stale.add(entry.entry_id)
            if old in self._auto_errors:
                self._auto_errors[entry.entry_id] = self._auto_errors[old]
        self._generation += 1
        self._queue_auto(owners.values())
        return keys

    def _release_display(self, entry: ImageToolEntry) -> None:
        if self._revisions[entry.revision_id].calculation and any(
            other.entry_id != entry.entry_id and other.revision_id == entry.revision_id
            for other in self._entries.values()
        ):
            return
        self._displayed.pop(entry.revision_id)

    def remove_entry(self, entry_id: str) -> None:
        """Close an entry. Existing results keep their recorded inputs."""
        self._write_generation()
        entry = self._entries[entry_id]
        self._index_sources(entry_id, set())
        if self._followers.get(entry_id):
            # Following this closed window now uses the last revision it displayed.
            roots = self._entry_roots[entry_id]
            self._protect(roots)
            self._closed[entry_id] = entry, roots
        self._release_display(entry)
        del self._entries[entry_id], self._entry_roots[entry_id]
        self._auto_pending.pop(entry_id, None)
        self._auto_errors.pop(entry_id, None)
        self._stale.discard(entry_id)
        self._generation += 1

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
            recipe = self._bind_prior(recipe, entry_id, self._prior_input(current))
        return self._run(entry_id, recipe, is_cancelled=is_cancelled)

    def _prior_input(self, current: ImageToolEntry) -> EarlierStep | Pin:
        return self._revision_input(current.revision_id, current.recipe)

    def _revision_input(
        self, revision: str, recipe: Recipe | FileLoad | None
    ) -> EarlierStep | Pin:
        if recipe is None:
            return Pin(revision)
        if recipe is self._revisions[revision].recipe:
            return EarlierStep(revision)
        return EarlierStep(revision, PendingRecipe(recipe))

    def _bind_prior(self, recipe: Recipe, entry_id: str, prior: Binding) -> Recipe:
        if not any(
            isinstance(binding, Follow) and binding.entry_id == entry_id
            for _, binding in recipe._captured.inputs
        ):
            return recipe
        inputs = []
        for slot, binding in recipe._captured.inputs:
            if isinstance(binding, Follow) and binding.entry_id == entry_id:
                if (
                    binding.source
                    and entry_id in self
                    and self._entries[entry_id].source_revision is not None
                ):
                    binding = self._unfiltered_input(self._entries[entry_id])
                else:
                    binding = prior
            inputs.append((slot, binding))
        return recipe._with_inputs(tuple(inputs))

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

    def set_auto_paused(self, paused: bool) -> None:
        """Retain pending automatic updates while a close decision is in progress."""
        self._write_generation()
        self._auto_paused = paused
        if not paused:
            self._schedule_pending_auto()

    def set_recipe(self, entry_id: str, recipe: Recipe | FileLoad) -> None:
        """Change the next calculation without changing the displayed revision."""
        self._write_generation()
        current = self._entries[entry_id]
        recipe = copy.deepcopy(recipe)
        sources = self._check_recipe(entry_id, recipe)
        self._index_sources(entry_id, sources)
        pinned = tuple(self._revisions[node] for node in self._recipe_revisions(recipe))
        self._protect(pinned)
        self._entry_roots[entry_id] = (
            self._revisions[current.revision_id],
            *(
                (self._revisions[current.source_revision],)
                if current.source_revision
                else ()
            ),
            *pinned,
        )
        self._entries[entry_id] = replace(current, recipe=recipe)
        self._auto_errors.pop(entry_id, None)
        self._stale.add(entry_id)
        candidates = self._invalidate({entry_id})
        self._generation += 1
        self._queue_auto([entry_id, *candidates])

    def plan_update(
        self,
        entry_id: str,
        *,
        command: typing.Literal["refresh", "reload"],
        include_operations: bool = False,
    ) -> UpdatePlan:
        """Preview the same request that Refresh or Reload would execute."""
        if command not in {"refresh", "reload"}:
            raise ValueError("Expected refresh or reload")
        entry = self._entries[entry_id]
        return self._plan(Follow(entry_id), recipe=entry.recipe, mode=command).describe(
            self, include_operations=include_operations
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
        self, entry_id: str, *, is_cancelled: Callable[[], bool] | None = None
    ) -> str:
        """Rebuild followed inputs from their origins. Pinned revisions stay fixed."""
        return self._run(
            entry_id,
            self._entries[entry_id].recipe,
            mode="reload",
            is_cancelled=is_cancelled,
        )

    def edit_operation(
        self,
        entry_id: str,
        *,
        displayed_revision: str,
        edited_revision: str,
        operations: Sequence[ToolProvenanceOperation],
        group_prefix: Sequence[str] = (),
        input_changes: Mapping[str, Binding] | None = None,
    ) -> str:
        """Edit a recorded application and publish its rebuilt branch."""
        recipe = self.edited_operation_recipe(
            entry_id,
            displayed_revision=displayed_revision,
            edited_revision=edited_revision,
            operations=operations,
            group_prefix=group_prefix,
            input_changes=input_changes,
        )
        if recipe is None:
            return displayed_revision
        return self._run(entry_id, recipe, mode="refresh", rebind=True)

    def edited_operation_recipe(
        self,
        entry_id: str,
        *,
        displayed_revision: str,
        edited_revision: str,
        operations: Sequence[ToolProvenanceOperation],
        group_prefix: Sequence[str] = (),
        input_changes: Mapping[str, Binding] | None = None,
    ) -> Recipe | FileLoad | None:
        """Replace a recorded operation or group and rebuild the selected branch.

        For a group, supply the last member as ``edited_revision`` and its preceding
        revision IDs in recorded order as ``group_prefix``.
        """
        if self._entries[entry_id].revision_id != displayed_revision:
            raise _SupersededEvaluation("The selected result changed while editing")
        if not operations:
            raise ValueError("An edit must supply a replacement operation")
        # The replacement consumes the first member's bindings. Downstream
        # applications receive its output in place of the last member's revision.
        first = edited_revision
        for preceding in reversed(group_prefix):
            producer = self._revisions[first].record.producer
            if not isinstance(producer, Application) or producer.inputs != (
                ("data", preceding),
            ):
                raise ValueError(
                    "A grouped edit requires contiguous recorded operations"
                )
            first = preceding
        original_recipe = self._revisions[first].recipe
        if not isinstance(original_recipe, Recipe):
            raise TypeError("The selected revision has no operation to edit")
        replacement_inputs = dict(original_recipe.inputs)
        if input_changes:
            if not input_changes.keys() <= replacement_inputs.keys():
                raise ValueError("Unknown input slot for the selected operation")
            replacement_inputs.update(input_changes)
        lineage = self.recorded_lineage(displayed_revision)
        if not any(record.revision_id == edited_revision for record in lineage):
            raise ValueError("The selected operation is not in this result's history")
        members = (*group_prefix, edited_revision)
        # Editors replace explicit arguments. Retain the calculation policy when
        # editing the corresponding operation, including an unchanged edit.
        if len(members) == len(operations):
            replacements = []
            for key, op in zip(members, operations, strict=True):
                applied = self._revisions[key].record.producer
                if (
                    isinstance(applied, Application)
                    and type(applied.operation) is type(op)
                    and getattr(applied.operation, "execution_options", None)
                    and not op.execution_options
                ):
                    op = op.model_copy(
                        update={
                            "execution_options": applied.operation.execution_options
                        }
                    )
                replacements.append(op)
            operations = replacements
        if (
            tuple(replacement_inputs.items()) == original_recipe.inputs
            and len(members) == len(operations)
            and all(
                isinstance(
                    (producer := self._revisions[key].record.producer), Application
                )
                and type(producer.operation) is type(operation)
                # A shared setup step can belong to a dialog group only on
                # the edited branch. The group identifies the steps to compare.
                and _setting_key(producer.operation, exclude_group=bool(group_prefix))
                == _setting_key(operation, exclude_group=bool(group_prefix))
                for key, operation in zip(members, operations, strict=True)
            )
        ):
            return None
        recipe = Recipe(
            operations[0],
            tuple(replacement_inputs.items()),
            input_names=original_recipe.input_names,
            output_name=original_recipe.output_name,
            source_kind=original_recipe.source_kind,
            view_role=original_recipe.view_role,
        )
        for op in operations[1:]:
            recipe = Recipe.bind(op, EarlierStep(first, PendingRecipe(recipe)))
        return self._branch_recipe(
            displayed_revision, {edited_revision: recipe}, lineage
        )

    def edited_file_recipe(
        self,
        entry_id: str,
        *,
        displayed_revision: str,
        loads: Mapping[str, FileLoad],
    ) -> Recipe | FileLoad | None:
        if self.current_revision(entry_id) != displayed_revision:
            raise _SupersededEvaluation("The selected result changed while editing")
        lineage = self.recorded_lineage(displayed_revision)
        originals = {r.revision_id: r.producer for r in lineage}
        changed = {}
        for key, load in loads.items():
            original = originals.get(key)
            if not isinstance(original, FileLoad):
                raise TypeError("Expected a recorded file input of the selected result")
            if original.source != load.source or _setting_key(
                original._captured
            ) != _setting_key(load._captured):
                changed[key] = load
        return (
            self._branch_recipe(displayed_revision, changed, lineage)
            if changed
            else None
        )

    def arranged_operation_recipe(
        self,
        entry_id: str,
        *,
        displayed_revision: str,
        revisions: Sequence[str],
        order: Sequence[int],
        identity: ToolProvenanceOperation,
    ) -> Recipe | FileLoad | None:
        """Reorder or remove applications while retaining their named side inputs."""
        if self.current_revision(entry_id) != displayed_revision:
            raise _SupersededEvaluation("The selected result changed while editing")
        if not revisions or len(set(revisions)) != len(revisions):
            raise ValueError("Select a contiguous sequence of recorded operations")
        if len(set(order)) != len(order) or any(
            not 0 <= index < len(revisions) for index in order
        ):
            raise ValueError("The requested order contains invalid operation positions")
        lineage = self.recorded_lineage(displayed_revision)
        members = {record.revision_id for record in lineage}
        if not members.issuperset(revisions):
            raise ValueError("The selected operation is not in this result's history")
        steps = []
        for index, key in enumerate(revisions):
            stored = self._revisions[key]
            if not isinstance(stored.recipe, Recipe) or not isinstance(
                stored.record.producer, Application
            ):
                raise TypeError("Expected a recorded operation")
            if index and stored.record.producer.inputs[0][1] != revisions[index - 1]:
                raise ValueError("The operations must share one primary input path")
            steps.append(stored.recipe)
        if tuple(order) == tuple(range(len(steps))):
            return None
        primary = steps[0].inputs[0][1]
        for index in order:
            step = steps[index]._with_inputs(
                (("data", primary), *steps[index].inputs[1:])
            )
            primary = EarlierStep(revisions[index], PendingRecipe(step))
        replacement = step if order else primary
        return self._branch_recipe(
            displayed_revision,
            {revisions[-1]: replacement},
            lineage,
            identity=identity,
        )

    def _branch_recipe(
        self,
        displayed: str,
        replacements: Mapping[str, Recipe | FileLoad | Binding],
        lineage: Sequence[DataRevision],
        *,
        identity: ToolProvenanceOperation | None = None,
    ) -> Recipe | FileLoad:
        edits: dict[str, Binding] = {
            target: EarlierStep(target, PendingRecipe(replacement))
            if isinstance(replacement, Recipe | FileLoad)
            else replacement
            for target, replacement in replacements.items()
        }
        for record in lineage:
            key = record.revision_id
            producer = record.producer
            if key in replacements or not isinstance(producer, Application):
                continue
            if any(source in edits for _, source in producer.inputs):
                original = typing.cast("Recipe", self._revisions[key].recipe)
                inputs = tuple(
                    (slot, edits.get(source, binding))
                    for (slot, binding), (_, source) in zip(
                        original.inputs, producer.inputs, strict=True
                    )
                )
                edits[key] = EarlierStep(
                    key, PendingRecipe(original._with_inputs(inputs))
                )
        result = edits[displayed]
        if isinstance(result, EarlierStep) and result.pending is not None:
            recipe = result.pending.recipe
        elif identity is not None:
            recipe = Recipe.bind(identity, result)
        else:
            raise ValueError("The replacement has no producing operation")
        return recipe

    def apply_operation_edits(
        self, updates: Mapping[str, tuple[str, Recipe | FileLoad]]
    ) -> None:
        """Evaluate all selected edits before installing any of their results."""
        if not updates:
            return
        for uid, (displayed, _) in updates.items():
            if self.current_revision(uid) != displayed:
                raise _SupersededEvaluation("A selected result changed while editing")
        first, *others = updates
        self._run(
            first,
            updates[first][1],
            mode="refresh",
            rebind=True,
            additional_edits={uid: updates[uid][1] for uid in others},
            additional_roots=tuple(Follow(uid) for uid in others),
        )

    def local_operations(self, revision: str) -> tuple[tuple[str, Application], ...]:
        """Read the local primary chain, stopping at its first bound input."""
        result = []
        while isinstance(
            (stored := self._revisions[revision]).record.producer, Application
        ):
            applied = stored.record.producer
            result.append((revision, copy.deepcopy(applied)))
            if not isinstance(stored.recipe, Recipe) or not isinstance(
                stored.recipe.inputs[0][1], EarlierStep
            ):
                break
            revision = applied.inputs[0][1]
        return tuple(reversed(result))

    def edit_displayed(
        self,
        edits: Mapping[str, DisplayedEdit],
        *,
        identity: ToolProvenanceOperation,
    ) -> None:
        """Apply local edits to recorded inputs and install all results together."""
        if not edits:
            return
        with self._displayed_transaction() as result:
            for uid, edit in edits.items():
                if self.current_revision(uid) != edit.displayed:
                    raise _SupersededEvaluation(
                        "The selected result changed while editing"
                    )
                self._stage_displayed_edit(uid, edit, result, identity)

    @contextmanager
    def _displayed_transaction(
        self,
        execution_scope: Callable[[], AbstractContextManager[Callable[[], None]]]
        | None = None,
        *,
        next_recipe: Recipe | FileLoad | None = None,
    ) -> Iterator[_Evaluation]:
        """Publish edited or restored values only after all windows accept them."""
        generation = self._write_generation()
        result = _Evaluation()
        sources = {}
        with ExitStack() as request:
            scope = execution_scope or self._execution_scope
            require_host = request.enter_context(scope()) if scope else None

            def require_current() -> None:
                if require_host is not None:
                    require_host()
                if generation != self._generation:
                    raise _SupersededEvaluation("Provenance changed during the edit")

            yield result
            for uid in result.entries:
                key, value = result.entries[uid]
                if self._validate_update:
                    self._validate_update(uid, value)
                sources[uid] = self._sources(
                    next_recipe or result.revisions[key].recipe,
                    revisions=ChainMap(result.revisions, self._revisions),
                )
            tuple(TopologicalSorter({**self._upstream, **sources}).static_order())
            require_current()
            first = next(iter(result.entries))
            self._publish(
                first,
                result,
                result.revisions[result.entries[first][0]].recipe,
                sources[first],
                "displayed",
                False,
                None,
                changed_sources=sources,
                require_current=require_current,
                restore=True,
                next_recipe=next_recipe,
                finish_request=request.close,
            )

    def _stage_displayed_edit(
        self,
        uid: str,
        edit: DisplayedEdit,
        result: _Evaluation,
        identity: ToolProvenanceOperation,
    ) -> None:
        # An application records the revision actually used for each slot. Its
        # recipe separately records where Refresh should get that slot next time.
        chain = self.local_operations(edit.displayed)
        positions = {key: index for index, (key, _) in enumerate(chain)}
        if not edit.replacements.keys() <= positions.keys():
            raise ValueError("An edited step is outside the selected calculation")
        before = chain[0][1].inputs[0][1] if chain else edit.displayed
        insertion = (
            0 if edit.append_after == before else positions[edit.append_after] + 1
        )
        start = min((positions[key] for key in edit.replacements), default=insertion)
        start = min(start, insertion)
        prior: Binding
        if start == len(chain):
            prior = self._prior_input(self._entries[uid])
            primary = edit.displayed
        else:
            key, applied = chain[start]
            primary = applied.inputs[0][1]
            prior = typing.cast("Recipe", self._revisions[key].recipe).inputs[0][1]
            if isinstance(prior, EarlierStep) and prior.revision_id != primary:
                prior = EarlierStep(primary)

        def apply(step: Recipe, edges: tuple[tuple[str, str], ...]) -> None:
            nonlocal primary, prior
            for key in {
                *self._recipe_revisions(step),
                *(source for _, source in edges),
            }:
                if key not in result.revisions:
                    result.sources[key] = self._revisions[key]
            for _, key in edges:
                if Pin(key) not in result.values:
                    result.values[Pin(key)] = key, self.data(key)
            self._evaluate_step(result, Follow(uid), step, None, edges=edges)
            primary, data = result.values[Follow(uid)]
            result.values[Pin(primary)] = primary, data
            prior = EarlierStep(primary)

        for index in range(start, len(chain) + 1):
            if index == insertion:
                for operation in edit.append:
                    apply(Recipe.bind(operation, prior), (("data", primary),))
            if index == len(chain):
                break
            key, applied = chain[index]
            operation = edit.replacements.get(key, applied.operation)
            if operation is not None:
                original = typing.cast("Recipe", self._revisions[key].recipe)
                step = Recipe(
                    operation,
                    (("data", prior), *original.inputs[1:]),
                    input_names=original.input_names,
                    output_name=original.output_name,
                    source_kind=original.source_kind,
                    view_role=original.view_role,
                )
                apply(step, (("data", primary), *applied.inputs[1:]))
        if primary not in result.revisions:
            # Removing the only operation still needs a result owned by this window.
            apply(Recipe.bind(identity, prior), (("data", primary),))
        result.entries[uid] = result.values[Pin(primary)]

    def restore_revisions(
        self,
        targets: Mapping[str, tuple[str, str]],
        *,
        prefixes: Mapping[str, Recipe] | None = None,
    ) -> None:
        """Restore the selected historical values and bindings in one publication."""
        if not targets:
            return
        with self._displayed_transaction() as result:
            for uid, (displayed, selected) in targets.items():
                if self.current_revision(uid) != displayed:
                    raise _SupersededEvaluation(
                        "The selected result changed before restoration"
                    )
                if not any(
                    record.revision_id == selected
                    for record in self.recorded_lineage(displayed)
                ):
                    raise ValueError(
                        "This revision is outside the selected result's history"
                    )
                stored = self._revisions[selected]
                # Two windows can restore the same input. Give each displayed
                # value its own revision while retaining its historical edges.
                key = uuid.uuid4().hex
                record = replace(stored.record, revision_id=key, owner_id=uid)
                prefix = (prefixes or {}).get(uid)
                if prefix is None:
                    value = self.data(selected)
                    restored = replace(stored, record=record)
                else:
                    applied = record.producer
                    if not isinstance(applied, Application) or not isinstance(
                        stored.recipe, Recipe
                    ):
                        raise TypeError(
                            "Only a recorded calculation can be restored to a prefix"
                        )
                    if prefix.inputs != stored.recipe.inputs:
                        raise ValueError(
                            "A restored prefix must use the recorded inputs"
                        )
                    producer = Application(prefix.operation, applied.inputs)
                    value = self._apply_operation(
                        producer,
                        {source: self.data(source) for _, source in applied.inputs},
                    )
                    restored = replace(
                        stored,
                        record=replace(record, producer=producer, checkpointed=True),
                        checkpoint=capture_display_data(value),
                        producer_id=key,
                        recipe=prefix,
                    )
                    value = typing.cast("xr.DataArray", restored.checkpoint)
                result.entries[uid] = key, value
                result.revisions[key] = restored

    def reload_entries(
        self,
        entries: Sequence[str],
        *,
        destinations: Callable[
            [Mapping[str, xr.DataArray]], AbstractContextManager[Mapping[str, str]]
        ]
        | None = None,
    ) -> None:
        """Reload one Manager selection in a shared, atomic request."""
        roots = tuple(
            dict.fromkeys(
                Follow(uid) for uid in entries if self._entries[uid].recipe is not None
            )
        )
        if roots:
            first = roots[0].entry_id
            self._run(
                first,
                self._entries[first].recipe,
                mode="reload",
                additional_roots=roots[1:],
                destinations=destinations,
            )

    def _sources(
        self,
        recipe: Recipe | FileLoad | None,
        *,
        revisions: Mapping[str, _StoredRevision] | None = None,
        allow_missing: bool = False,
        displayed_only: bool = False,
    ) -> set[str]:
        """Find entries followed directly or through an earlier producing step."""
        revisions = self._revisions if revisions is None else revisions
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
                            not allow_missing
                            and binding.entry_id not in self._entries
                            and binding.entry_id not in self._closed
                        ):
                            raise KeyError(binding.entry_id)
                        if not displayed_only or not binding.source:
                            sources.add(binding.entry_id)
                    elif isinstance(binding, Pin):
                        if binding.revision_id not in revisions:
                            raise KeyError(binding.revision_id)
                    elif isinstance(binding, EarlierStep):
                        if binding not in visited:
                            pending.append(self._earlier_recipe(binding, revisions))
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

    def _earlier_recipe(
        self, binding: EarlierStep, revisions: Mapping[str, _StoredRevision]
    ) -> Recipe | FileLoad:
        stored = revisions[binding.revision_id]
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
        return self._earlier_steps_changed(pending)

    def _earlier_steps_changed(self, pending: list[tuple[EarlierStep, str]]) -> bool:
        visited: set[tuple[EarlierStep, str]] = set()
        while pending:
            binding, node = pending.pop()
            if (binding, node) in visited:
                continue
            visited.add((binding, node))
            stored = self._revisions[node]
            step = stored.recipe
            if binding.pending is not None and binding.revision_id == node:
                next_step = binding.pending.recipe
                if (
                    not isinstance(step, Recipe)
                    or not isinstance(next_step, Recipe)
                    or next_step._captured.operation is not step._captured.operation
                ):
                    return True
                # A closed or replaced source can acquire a different binding
                # that still supplies exactly the input used in this calculation.
                step = next_step
            if not isinstance(step, Recipe):
                continue
            applied = typing.cast("Application", stored.record.producer)
            for (_, source), (_, used) in zip(
                step._captured.inputs, applied.inputs, strict=True
            ):
                if isinstance(source, Pin) and source.revision_id != used:
                    return True
                if isinstance(source, Follow) and (
                    (
                        source.entry_id not in self._entries
                        and source.entry_id not in self._closed
                    )
                    or self.followed_revision(source) != used
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

    def _invalidate(
        self, changed: set[str], *, display_only: set[str] | None = None
    ) -> set[str]:
        """Mark followers stale and collect Auto candidates along the affected paths."""
        pending = list(changed)
        candidates: set[str] = set()
        while pending:
            source = pending.pop()
            for child in self._followers.get(source, ()):
                if (
                    display_only
                    and source in display_only
                    and source
                    not in self._sources(
                        self._entries[child].recipe,
                        allow_missing=True,
                        displayed_only=True,
                    )
                ):
                    continue
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
        sources = self.follows(entry_id)
        return (sources & self._stale) | sources.difference(self._entries, self._closed)

    def _queue_auto(self, candidates: Iterable[str]) -> None:
        for entry_id in candidates:
            if (
                entry_id in self._entries
                and self._entries[entry_id].auto
                and entry_id in self._stale
            ):
                self._auto_pending[entry_id] = None
        self._schedule_pending_auto()

    def _schedule_pending_auto(self) -> None:
        if (
            self._auto_paused
            or self._auto_scheduled
            or self._posting_auto
            or not self._auto_pending
        ):
            return
        # An inline scheduler finishes the pass before returning. Post its
        # continuation iteratively; an event-loop scheduler posts one callback.
        self._posting_auto = True
        try:
            while (
                self._auto_pending
                and not self._auto_scheduled
                and not self._auto_paused
            ):
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
            while self._auto_pending and not self._auto_paused:
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
        self._check_cycle(entry_id, sources)
        return sources

    def _check_cycle(self, entry_id: str, sources: set[str]) -> None:
        if sources and (
            entry_id in sources or not sources.isdisjoint(self._descendants([entry_id]))
        ):
            raise ValueError("Refresh bindings would create a cycle")

    def _index_sources(self, entry_id: str, sources: set[str]) -> None:
        old_sources = self._upstream.pop(entry_id, frozenset())
        if sources:
            self._upstream[entry_id] = frozenset(sources)
        for source in old_sources - sources:
            self._followers[source].remove(entry_id)
            if not self._followers[source]:
                del self._followers[source]
                self._closed.pop(source, None)
        for source in sources - old_sources:
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
        auto: bool = False,
        parent_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        additional_roots: Sequence[Follow] = (),
        rebind: bool = False,
        additional_edits: Mapping[str, Recipe | FileLoad] | None = None,
        destinations: Callable[
            [Mapping[str, xr.DataArray]], AbstractContextManager[Mapping[str, str]]
        ]
        | None = None,
    ) -> str:
        generation = self._write_generation()
        if recipe is None and mode != "displayed":
            _check_cancelled(is_cancelled)
            return self._entries[entry_id].revision_id
        sources = (
            self._check_recipe(entry_id, recipe)
            if mode == "displayed" or rebind
            else self._sources(recipe)
        )
        changed_sources = {
            uid: self._check_recipe(uid, step)
            for uid, step in (additional_edits or {}).items()
        }

        def check_batch() -> None:
            if changed_sources:
                tuple(
                    TopologicalSorter(
                        {**self._upstream, entry_id: sources, **changed_sources}
                    ).static_order()
                )

        check_batch()
        root = Follow(entry_id)
        plan = self._plan(
            root,
            recipe=recipe,
            mode=mode,
            additional_roots=additional_roots,
            additional_edits=additional_edits,
        )
        with ExitStack() as request:
            require_host = (
                request.enter_context(self._execution_scope())
                if self._execution_scope is not None
                else None
            )
            try:
                result = self._evaluate(plan, is_cancelled=is_cancelled)
                if self._validate_update is not None:
                    for uid, (_, data) in result.entries.items():
                        self._validate_update(uid, data)
            except _SupersededEvaluation:
                raise
            except Exception:
                # An operation can fail on an input that changed while it ran.
                if self._generation != generation:
                    self._check_current(plan)
                raise

            def require_current() -> None:
                if require_host is not None:
                    require_host()
                _check_cancelled(is_cancelled)
                if self._generation != generation:
                    self._check_current(plan)
                    if mode == "displayed" or rebind:
                        # A new edge elsewhere may have closed a proposed live cycle.
                        self._check_recipe(entry_id, recipe)
                        check_batch()

            require_current()
            if destinations is not None:
                redirects = request.enter_context(
                    destinations(
                        {uid: value for uid, (_, value) in result.entries.items()}
                    )
                )
                require_current()
                if len(set(redirects.values())) != len(redirects):
                    raise ValueError("Each new result requires its own ImageTool")
                for previous, new in redirects.items():
                    self._check_new_entry(new, None)
                    key, value = result.entries.pop(previous)
                    stored = result.revisions[key]
                    # An open-as-new reload publishes the same calculation under
                    # the new entry. The previous window keeps its existing value.
                    fresh = uuid.uuid4().hex
                    result.revisions[fresh] = replace(
                        stored,
                        record=replace(stored.record, revision_id=fresh, owner_id=new),
                    )
                    result.entries[new] = fresh, value
                    changed_sources[new] = self._sources(stored.recipe)
            return self._publish(
                entry_id,
                result,
                recipe,
                sources,
                mode,
                auto,
                parent_id,
                rebind=rebind,
                changed_sources=changed_sources,
                require_current=require_current,
                finish_request=request.close,
            )

    def _publish(
        self,
        entry_id: str,
        result: _Evaluation,
        recipe: Recipe | FileLoad | None,
        sources: set[str],
        mode: _Mode,
        auto: bool,
        parent_id: str | None,
        *,
        rebind: bool = False,
        next_recipe: Recipe | FileLoad | None = None,
        require_current: Callable[[], None] | None = None,
        finish_request: Callable[[], None] | None = None,
        changed_sources: Mapping[str, set[str]] | None = None,
        restore: bool = False,
        entry_recipe: Recipe | None = None,
        output_name: str | None = None,
        prepare_view: _PrepareView | None = None,
    ) -> str:
        # Commit the prepared values and edges together after every operation succeeds.
        display_only: set[str] = set()
        # Provisional revisions remain reachable from the evaluation while the host
        # installs the complete batch. Entries still point at their old results. The
        # weak revision index releases provisional records if installation fails.
        self._revisions.update(result.revisions)
        if prepare_view is not None:
            # A record-only publication can include the window's actual layout
            # and filter. Keep the source provisional until the view accepts it.
            key, source = result.entries[entry_id]
            prior = self._revision_input(key, next_recipe or entry_recipe or recipe)
            generation = self._generation
            view = prepare_view(entry_id, prior, source)
            if generation != self._generation:
                raise _SupersededEvaluation("Provenance changed during installation")
            if view:
                old_value = self._displayed.get(key)
                self._displayed[key] = source
                try:
                    publish_view = self.prepare_evaluation(
                        entry_id,
                        view,
                        auto=auto,
                        parent_id=parent_id,
                    )
                finally:
                    if old_value is None:
                        self._displayed.pop(key, None)
                    else:
                        self._displayed[key] = old_value
                return publish_view()
        # Record-only calls already installed the data in their owning window.
        with ExitStack() as batch:
            if self._install is not None and require_current is not None:
                try:
                    materialized = batch.enter_context(
                        self._install(
                            {uid: pair[0] for uid, pair in result.entries.items()},
                            dict((*result.values.values(), *result.entries.values())),
                        )
                    )
                except Exception:
                    # An installation error on superseded inputs does not belong
                    # to the newer request. Auto can retry the current inputs.
                    require_current()
                    raise
                # Synchronous host callbacks may update our inputs. Recheck while
                # every staged window is still rollbackable.
                require_current()
                checkpoints = []
                for key, data in materialized.items():
                    stored = result.revisions.get(key) or result.sources.get(key)
                    if stored is None or data.chunks is not None:
                        raise ValueError("The host returned an invalid materialization")
                    checkpoints.append((stored, data))
                for stored, data in checkpoints:
                    # The host owns a separate window buffer. Keep the values it
                    # actually computed, so later changes to a lazy input cannot
                    # alter an already materialized result or intermediate step.
                    stored.checkpoint = data
                    stored.record = replace(stored.record, checkpointed=True)
                    stored.depends_on_lazy_input = False
                for uid, (key, data) in result.entries.items():
                    data = materialized.get(key, data)
                    if restore and self._installed_step is not None:
                        # ImageTool can cast the result or keep a compatible axis
                        # layout. Record those operations while rollback is possible.
                        while (
                            adjustment := self._installed_step(uid, key, data)
                        ) is not None:
                            if uid == entry_id and next_recipe is not None:
                                # Refresh must apply the same display conversion to
                                # the pending result, including a change of source.
                                adjustment = adjustment._with_inputs(
                                    (
                                        (
                                            "data",
                                            EarlierStep(
                                                key, PendingRecipe(next_recipe)
                                            ),
                                        ),
                                    )
                                )
                                next_recipe = adjustment
                            result.values[Pin(key)] = key, data
                            self._evaluate_step(
                                result,
                                Follow(uid),
                                adjustment,
                                None,
                                edges=(("data", key),),
                            )
                            key, data = result.values[Follow(uid)]
                            self._revisions[key] = result.revisions[key]
                    result.entries[uid] = key, data
                require_current()
            for key, value in result.view_inputs.items():
                base = self._revisions[key]
                base.checkpoint = value
                base.record = replace(base.record, checkpointed=True)
            for target, (revision_id, data) in result.entries.items():
                old = self._entries.get(target)
                stored = result.revisions[revision_id]
                effective = (
                    next_recipe
                    if next_recipe is not None and target == entry_id
                    else entry_recipe
                    if target == entry_id and entry_recipe is not None
                    else stored.recipe
                )
                old_source = (
                    self.followed_revision(Follow(target, source=True))
                    if old is not None
                    else None
                )
                if old is None:
                    entry = ImageToolEntry(
                        target,
                        revision_id,
                        effective,
                        auto,
                        parent_id,
                        len(self._entries),
                    )
                else:
                    self._release_display(old)
                    entry = replace(old, revision_id=revision_id, recipe=effective)
                if target == entry_id and (mode == "displayed" or rebind):
                    self._index_sources(target, sources)
                    if mode == "displayed":
                        entry = replace(entry, output_name=output_name)
                elif changed_sources and target in changed_sources:
                    self._index_sources(target, changed_sources[target])
                source_view = self._unfiltered_revision(
                    target, stored, result, old_source
                )
                entry = replace(
                    entry,
                    source_revision=source_view.record.revision_id
                    if source_view
                    else None,
                )
                self._entries[target] = entry
                if old_source == self.followed_revision(Follow(target, source=True)):
                    display_only.add(target)
                self._displayed[revision_id] = data
                self._entry_roots[target] = (
                    (stored, *result.sources.values())
                    if next_recipe is not None or entry_recipe is not None
                    else (stored,)
                )
                if source_view is not None:
                    self._entry_roots[target] += (source_view,)
                self._auto_errors.pop(target, None)

            candidates = self._invalidate(
                set(result.entries), display_only=display_only
            )
            self._stale.difference_update(result.entries)
            if next_recipe is not None or (
                mode == "displayed"
                and recipe is not None
                and (sources & self._stale or self._earlier_input_changed(recipe))
            ):
                # Derive and Replace consume the data on screen, including stale data.
                self._stale.add(entry_id)
                candidates.add(entry_id)
            if restore:
                for uid, (key, _) in result.entries.items():
                    if (
                        self._earlier_steps_changed([(EarlierStep(key), key)])
                        or self._upstream.get(uid, frozenset()) & self._stale
                    ):
                        self._stale.add(uid)
                        candidates.add(uid)
            self._generation += 1
        if finish_request is not None:
            # Retain the sources of this commit before observers start other work.
            finish_request()
        if self._published is not None:
            self._published(tuple(result.entries))
        self._queue_auto(candidates)
        return self._entries[entry_id].revision_id

    def _plan_input(
        self, binding: Binding, *, mode: _Mode, revisions: Mapping[str, _StoredRevision]
    ) -> _Action:
        if isinstance(binding, Follow):
            if binding.entry_id in self._closed:
                return Pin(self.followed_revision(binding))
            entry = self._entries[binding.entry_id]
            if entry.recipe is not None and (
                mode == "reload"
                or (mode == "refresh" and binding.entry_id in self._stale)
            ):
                if binding.source:
                    if (
                        entry.source_revision is not None
                        and isinstance(entry.recipe, Recipe)
                        and entry.recipe.view_role is not None
                    ):
                        return self._unfiltered_input(entry)
                    return Follow(entry.entry_id)
                return entry.recipe
            return Pin(self.followed_revision(binding))
        if isinstance(binding, EarlierStep):
            if mode != "displayed":
                return self._earlier_recipe(binding, revisions)
            return Pin(binding.revision_id)

        stored = revisions[binding.revision_id]
        data = self._displayed.get(binding.revision_id, stored.checkpoint)
        if data is not None and not (
            stored.depends_on_lazy_input and stored.checkpoint is None
        ):
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
        revisions: Mapping[str, _StoredRevision] | None = None,
        additional_roots: Sequence[Follow] = (),
        additional_edits: Mapping[str, Recipe | FileLoad] | None = None,
    ) -> _Plan:
        if revisions is None:
            revisions = self._revisions
        plan: dict[Binding, _Action] = {}
        producers: dict[str, list[Follow | EarlierStep]] = {}
        producer_ids: dict[Binding, str] = {}
        dependencies: dict[Binding, tuple[Binding, ...]] = {}
        sources: dict[str, _StoredRevision] = {}
        observed: dict[str, tuple[ImageToolEntry | None, bool]] = {}
        if recipe is not None:
            # Pending edits can pin inputs that Replace does not evaluate.
            sources.update(
                (node, revisions[node]) for node in self._recipe_revisions(recipe)
            )
        for extra in (additional_edits or {}).values():
            sources.update(
                (key, revisions[key]) for key in self._recipe_revisions(extra)
            )
        pending: list[Binding] = [*reversed(additional_roots), root]
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
                sources[binding.revision_id] = revisions[binding.revision_id]
            step = (
                recipe
                if binding == root and recipe is not None
                else additional_edits[binding.entry_id]
                if additional_edits
                and isinstance(binding, Follow)
                and not binding.source
                and binding.entry_id in additional_edits
                else self._plan_input(binding, mode=mode, revisions=revisions)
            )
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
                producer_id = revisions[revision_id].producer_id
                producer_ids[binding] = producer_id
                producers.setdefault(producer_id, []).append(binding)
            plan[binding] = step
            inputs: tuple[Binding, ...] = ()
            if isinstance(step, Binding):
                inputs = (step,)
            elif isinstance(step, Recipe):
                inputs = tuple(source for _, source in step._captured.inputs)
            if (
                isinstance(binding, Follow)
                and binding.source
                and binding.entry_id in self._entries
                and (
                    mode == "reload"
                    or (mode == "refresh" and binding.entry_id in self._stale)
                )
            ):
                # Rebuild the window as well as the unfiltered value consumed here.
                inputs = (*inputs, Follow(binding.entry_id))
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
        )

    def _evaluate(
        self, plan: _Plan, *, is_cancelled: Callable[[], bool] | None = None
    ) -> _Evaluation:
        self._protect(
            plan.sources.values(), require_current=lambda: self._check_current(plan)
        )
        # A plan preview only inspects IDs. Resolve the sealed payloads at the
        # start of execution, before any producer can reenter the Manager.
        pinned = {
            step[0]: self._displayed.get(step[0], plan.sources[step[0]].checkpoint)
            for _, step in plan.steps
            if isinstance(step, tuple)
        }
        result = _Evaluation(sources=plan.sources)
        generation = self._generation
        for binding, step in plan.steps:
            _check_cancelled(is_cancelled)
            if self._generation != generation:
                self._check_current(plan)
                generation = self._generation
            if isinstance(step, tuple):
                value = pinned[step[0]]
                result.values[binding] = (step[0], step[1] if value is None else value)
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
            if (
                isinstance(binding, Follow)
                and not binding.source
                and not isinstance(step, Pin)
            ):
                # A live entry publishes even if an earlier step ran its recipe.
                result.entries[binding.entry_id] = result.values[binding]
        return result

    def _evaluate_step(
        self,
        result: _Evaluation,
        binding: Binding,
        step: Recipe | FileLoad,
        producer_id: str | None,
        *,
        edges: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        producer: Application | FileLoad
        if isinstance(step, FileLoad):
            if self._read_file is None:
                raise ValueError(f"No file reader for {step.source.path!r}")
            data = capture_display_data(
                self._read_file(step.source, copy.deepcopy(step._captured)),
                source_path=step.source.path,
            )
            producer = step
        else:
            inputs = (
                edges
                if edges is not None
                else tuple(
                    (slot, result.values[source][0])
                    for slot, source in step._captured.inputs
                )
            )
            producer = Application(step._captured.operation, inputs)
            values = (
                dict(result.values[Pin(key)] for _, key in edges)
                if edges is not None
                else dict(result.values[source] for _, source in step._captured.inputs)
            )
            data = self._apply_operation(
                producer,
                {
                    node: value.copy(
                        deep=not getattr(
                            step._captured.operation, "reads_input_only", False
                        )
                    )
                    for node, value in values.items()
                },
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
        predecessors = tuple(
            result.revisions.get(node) or result.sources[node] for node in parent_ids
        )
        lazy_input = any(parent.depends_on_lazy_input for parent in predecessors)
        checkpoint = (
            data
            if isinstance(step, FileLoad) or (lazy_input and data.chunks is None)
            else None
        )
        result.revisions[revision_id] = _StoredRevision(
            DataRevision(
                revision_id,
                producer,
                checkpoint is not None,
                binding.entry_id
                if isinstance(binding, Follow)
                else self._revisions[binding.revision_id].record.owner_id,
            ),
            revision_id if producer_id is None else producer_id,
            step,
            checkpoint,
            predecessors,
            depends_on_lazy_input=lazy_input
            if checkpoint is None
            else checkpoint.chunks is not None,
        )
