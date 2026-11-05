"""Data records for the ImageTool provenance graph."""

from __future__ import annotations

import copy
import keyword
import typing
from dataclasses import dataclass, field

import xarray as xr

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from erlab.interactive.imagetool._provenance._model import (
        FileReplayCall,
        ToolProvenanceOperation,
    )


def check_id(identifier: object) -> None:
    if (
        not isinstance(identifier, str)
        or not identifier.strip()
        or "/" in identifier
        or "\x00" in identifier
        or identifier in {".", ".."}
    ):
        raise ValueError(f"Invalid provenance ID {identifier!r}")


def checkpoint_data(data: xr.DataArray) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError("Revision data must be a DataArray")
    # A copied lazy array can still read from a mutable external source.
    return data.copy(deep=True).load()


@dataclass(frozen=True)
class Follow:
    """Use the revision displayed by this entry."""

    entry_id: str


@dataclass(frozen=True)
class Pin:
    """Use this exact data revision."""

    revision_id: str


@dataclass(frozen=True, eq=False)
class PendingRecipe:
    """A producer edit captured by Replace. Each snapshot is a distinct plan input."""

    recipe: Recipe | FileLoad


@dataclass(frozen=True)
class EarlierStep:
    """Consume this revision now. Refresh uses its captured edit, if any."""

    revision_id: str
    pending: PendingRecipe | None = None


Binding: typing.TypeAlias = Follow | Pin | EarlierStep


def _check_slots(operation: ToolProvenanceOperation, names: tuple[str, ...]) -> None:
    slots = ("data", *operation.input_slots())
    if (
        names != slots
        or len(set(slots)) != len(slots)
        or any(not slot.isidentifier() or keyword.iskeyword(slot) for slot in slots)
    ):
        raise ValueError("Bindings must match the ordered operation input slots")


_T = typing.TypeVar("_T")


@dataclass
class _Settings(typing.Generic[_T]):
    """Share a captured value until this owner hands out a writable reference."""

    value: _T
    exposed: bool = field(default=False, compare=False, repr=False)

    def editable(self) -> _T:
        if not self.exposed:
            self.value = copy.deepcopy(self.value)
            self.exposed = True
        return self.value

    def __deepcopy__(self, memo: dict[int, typing.Any]) -> _Settings[_T]:
        return _Settings(
            copy.deepcopy(self.value, memo) if self.exposed else self.value
        )


@dataclass(frozen=True)
class _RecipeData:
    operation: ToolProvenanceOperation
    inputs: tuple[tuple[str, Binding], ...]


@dataclass(frozen=True, init=False)
class Recipe:
    _settings: _Settings[_RecipeData]

    def __init__(
        self,
        operation: ToolProvenanceOperation,
        inputs: tuple[tuple[str, Binding], ...],
    ) -> None:
        _check_slots(operation, tuple(slot for slot, _ in inputs))
        object.__setattr__(
            self, "_settings", _Settings(copy.deepcopy(_RecipeData(operation, inputs)))
        )

    @property
    def _captured(self) -> _RecipeData:
        return self._settings.value

    @property
    def operation(self) -> ToolProvenanceOperation:
        """Return this recipe's editable operation, isolated from other snapshots."""
        return self._settings.editable().operation

    @property
    def inputs(self) -> tuple[tuple[str, Binding], ...]:
        bindings = self._captured.inputs
        if any(
            isinstance(b, EarlierStep) and b.pending is not None for _, b in bindings
        ):
            # Detach nested editable recipes before exposing their settings.
            return self._settings.editable().inputs
        return bindings

    def _with_inputs(self, inputs: tuple[tuple[str, Binding], ...]) -> Recipe:
        # Both callers use graph-owned operations and bindings.
        operation = self._captured.operation
        _check_slots(operation, tuple(slot for slot, _ in inputs))
        result = copy.copy(self)
        object.__setattr__(
            result, "_settings", _Settings(_RecipeData(operation, inputs))
        )
        return result

    @classmethod
    def bind(
        cls, operation: ToolProvenanceOperation, data: Binding, **inputs: Binding
    ) -> Recipe:
        slots = operation.input_slots()
        if set(inputs) != set(slots):
            raise ValueError("Bindings must match the operation input slots")
        return cls(
            operation, (("data", data), *((slot, inputs[slot]) for slot in slots))
        )


@dataclass(frozen=True)
class Application:
    """One calculation. Each ordered slot gives one edge to the result."""

    operation: ToolProvenanceOperation
    inputs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _check_slots(self.operation, tuple(slot for slot, _ in self.inputs))
        for _, node in self.inputs:
            check_id(node)

    def apply(self, values: Mapping[str, xr.DataArray]) -> xr.DataArray:
        (_, primary), *others = self.inputs
        data = values[primary]
        if others:
            return self.operation.apply_with_inputs(
                data, {slot: values[node] for slot, node in others}
            )
        return self.operation.apply(data)

    def code(self, names: Mapping[str, str], *, output_name: str) -> str:
        """Emit the applied step using names for its exact input revisions."""
        (_, primary), *others = self.inputs
        return self.operation.replay_code(
            names[primary],
            output_name=output_name,
            bound_inputs={slot: names[node] for slot, node in others},
            reserved_names=(*names.values(), output_name),
        )


@dataclass(frozen=True)
class FileSource:
    source_id: str
    path: str


@dataclass(frozen=True, init=False)
class FileLoad:
    """The file is a node. Loader settings and selection belong to the edge."""

    source: FileSource
    _settings: _Settings[FileReplayCall]

    def __init__(self, source: FileSource, call: FileReplayCall) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_settings", _Settings(copy.deepcopy(call)))

    @property
    def _captured(self) -> FileReplayCall:
        return self._settings.value

    @property
    def call(self) -> FileReplayCall:
        """Return this load's editable call, isolated from other snapshots."""
        return self._settings.editable()

    @property
    def inputs(self) -> tuple[tuple[str, str], ...]:
        return (("file", self.source.source_id),)


@dataclass(frozen=True)
class DataRevision:
    revision_id: str
    producer: Application | FileLoad | None
    checkpointed: bool


@dataclass(frozen=True)
class ImageToolEntry:
    entry_id: str
    revision_id: str
    recipe: Recipe | FileLoad | None
    auto: bool
    parent_id: str | None
    order: int
