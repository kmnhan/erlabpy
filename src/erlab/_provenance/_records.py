"""Data records for the ImageTool provenance graph."""

from __future__ import annotations

import copy
import keyword
import os
import typing
from dataclasses import dataclass, field

import numpy as np
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


class WorkspaceSourcePath(str):
    """Logical workspace location. Its reader owns the payload across saves."""


def _file_version(path: str) -> tuple[int, int, int, int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class _LazyReadGuard:
    versions: tuple[tuple[str, tuple[int, int, int, int, int] | None], ...]

    def problem(self) -> str | None:
        for path, version in self.versions:
            if version is None or _file_version(path) != version:
                return f"Recorded lazy input {path!r} is unavailable or has changed."
        return None


def _check_lazy_input(block, *, guard: _LazyReadGuard):
    # Dask checks the source again after reading each block, including any
    # change that occurs between previewing a command and running the read.
    if reason := guard.problem():
        raise OSError(reason)
    return block


def lazy_input_guards(data: xr.DataArray) -> tuple[_LazyReadGuard, ...]:
    """Return retained file versions without reading any chunk."""
    from dask.highlevelgraph import HighLevelGraph

    guards: dict[_LazyReadGuard, None] = {}
    for variable in (
        data.variable,
        *(coord.variable for coord in data.coords.values()),
    ):
        if variable.chunks is None:
            continue
        collection = variable.data.__dask_graph__()
        if isinstance(collection, HighLevelGraph):
            for layer in collection.layers.values():
                guard = (layer.annotations or {}).get("erlab_recorded_input")
                if isinstance(guard, _LazyReadGuard):
                    guards[guard] = None
    return tuple(guards)


def lazy_input_problem(data: xr.DataArray) -> str | None:
    """Check the same guards as the array's chunk tasks without reading any chunk."""
    for guard in lazy_input_guards(data):
        if reason := guard.problem():
            return reason
    return None


def guard_lazy_data(
    data: xr.DataArray, guards: typing.Iterable[_LazyReadGuard]
) -> xr.DataArray:
    """Apply recorded file checks to an existing computation, without replaying it."""
    import dask

    guards = tuple(dict.fromkeys(guards))
    if not guards:
        return data
    needed = []
    for variable in (data.variable, *(c.variable for c in data.coords.values())):
        existing = lazy_input_guards(xr.DataArray(variable))
        needed.append(
            tuple(guard for guard in guards if guard not in existing)
            if variable.chunks is not None
            else ()
        )
    if not any(needed):
        return data
    result = data.copy(deep=True)
    for variable, extra in zip(
        (result.variable, *(c.variable for c in result.coords.values())),
        needed,
        strict=True,
    ):
        if not extra:
            continue
        array = variable.data
        for guard in extra:
            with dask.annotate(erlab_recorded_input=guard):
                array = array.map_blocks(
                    _check_lazy_input,
                    guard=guard,
                    dtype=array.dtype,
                    meta=array._meta,
                )
        variable.data = array
    return result


def checkpoint_data(data: xr.DataArray) -> xr.DataArray:
    """Freeze an explicit checkpoint before the caller can change its source."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Revision data must be a DataArray")
    # A NumPy array backed by immutable bytes can be shared even if the sender
    # keeps its view. Read-only flags alone can be reset by a writeable owner.
    if data.variable._in_memory:
        buffer = data.data
        while isinstance(buffer, (np.ndarray, memoryview)):
            buffer = buffer.base if isinstance(buffer, np.ndarray) else buffer.obj
        if isinstance(buffer, bytes):
            return data.copy(deep=True, data=data.data).load()
    return data.copy(deep=True).load()


def own_eager_data(data: xr.DataArray) -> xr.DataArray:
    """Accept eager values before the caller can reuse its mutable allocation."""
    values = data.data
    if values.dtype.hasobject:
        return data.copy(deep=True)
    buffer = values
    while isinstance(buffer, (np.ndarray, memoryview)):
        buffer = buffer.base if isinstance(buffer, np.ndarray) else buffer.obj
    if not isinstance(buffer, bytes):
        values = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(
            values.shape
        )
    # Metadata and coordinate arrays remain isolated, including for shared values.
    return data.copy(deep=True, data=values)


def capture_display_data(
    data: xr.DataArray,
    *,
    source_path: str | None = None,
    live: bool = False,
) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError("Revision data must be a DataArray")
    variables = (data.variable, *(coord.variable for coord in data.coords.values()))
    if all(variable._in_memory for variable in variables):
        return own_eager_data(data) if live else checkpoint_data(data)

    # Copy the Dask expression, not its evaluated payload. Convert unchunked
    # backend arrays to chunked reads without indexing the complete variable.
    # A workspace save can append a generation while an open reader still
    # owns the same payload. Its logical path does not identify a file load.
    encoded_sources = (
        var.encoding.get("source") for var in variables if not var._in_memory
    )
    files = [
        path for path in encoded_sources if not isinstance(path, WorkspaceSourcePath)
    ]
    paths = {
        os.path.abspath(os.fsdecode(path))
        for path in (source_path, *files)
        if isinstance(path, (str, bytes, os.PathLike))
    }
    versions = tuple((path, _file_version(path)) for path in sorted(paths))
    result = (data if data.chunks is not None else data.chunk("auto")).copy(deep=True)
    if versions:
        result = guard_lazy_data(result, (_LazyReadGuard(versions),))
    return result


@dataclass(frozen=True)
class Follow:
    """Use the revision displayed by this entry."""

    entry_id: str
    source: bool = False


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
    input_names: tuple[str, ...]
    output_name: str | None
    source_kind: typing.Literal["full_data", "public_data", "selection"] | None
    view_role: typing.Literal["layout", "filter"] | None

    def __init__(
        self,
        operation: ToolProvenanceOperation,
        inputs: tuple[tuple[str, Binding], ...],
        *,
        view_role: typing.Literal["layout", "filter"] | None = None,
        input_names: tuple[str, ...] = (),
        output_name: str | None = None,
        source_kind: typing.Literal["full_data", "public_data", "selection"]
        | None = None,
    ) -> None:
        if input_names and len(input_names) != len(inputs):
            raise ValueError("Each named input must correspond to one slot")
        object.__setattr__(self, "view_role", view_role)
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "source_kind", source_kind)
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
        # Lightweight graph operations can implement apply without a model.
        with xr.set_options(**getattr(self.operation, "execution_options", {})):
            if others:
                return self.operation.apply_with_inputs(
                    data, {slot: values[node] for slot, node in others}
                )
            return self.operation.apply(data)

    def code(self, names: Mapping[str, str], *, output_name: str) -> str:
        """Emit the applied step using names for its exact input revisions."""
        (_, primary), *others = self.inputs
        statement = self.operation.replay_code(
            names[primary],
            output_name=output_name,
            bound_inputs={slot: names[node] for slot, node in others},
            reserved_names=(*names.values(), output_name),
        )
        return self.operation.execution_code(statement)


@dataclass(frozen=True)
class FileSource:
    source_id: str
    path: str


@dataclass(frozen=True)
class FileLoadPresentation:
    """Notebook names and labels retained when importing an existing file load."""

    label: str
    output_name: str
    seed_code: str | None
    loader_label: str
    loader_text: str
    kwargs_text: str
    load_code: str | None


@dataclass(frozen=True, init=False)
class FileLoad:
    """The file is a node. Loader settings and selection belong to the edge."""

    source: FileSource
    _settings: _Settings[FileReplayCall]

    presentation: FileLoadPresentation | None

    def __init__(
        self,
        source: FileSource,
        call: FileReplayCall,
        presentation: FileLoadPresentation | None = None,
    ) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "presentation", presentation)
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
    owner_id: str
    # Imported history with no verifiable input edge is descriptive. Only
    # the checkpoint supplies data for this revision.
    imported_history: str | None = None


@dataclass(frozen=True)
class ImageToolEntry:
    entry_id: str
    revision_id: str
    recipe: Recipe | FileLoad | None
    auto: bool
    parent_id: str | None
    order: int
    source_revision: str | None = None
    output_name: str | None = None
