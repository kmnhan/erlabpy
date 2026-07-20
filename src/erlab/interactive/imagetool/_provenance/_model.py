"""Helpers for recording, displaying, and replaying ImageTool provenance.

The saved provenance model stores enough history to explain or rebuild an ImageTool
without persisting runtime-only replay state. :class:`ToolProvenanceSpec` has a small
set of source forms:

1. Live single-parent source specs.

   - Use :func:`full_data` when the derived tool should begin from the parent
     ImageTool's current array exactly as shown to the caller.

   - Use :func:`public_data` when the derived tool should begin from the parent
     ImageTool's public data model, including restoration of non-uniform dimensions.

   - Use :func:`selection` when the derived tool should begin from the public
     ImageTool selection model, including restoration of non-uniform dimensions.

2. Durable replay specs.

   - Use :func:`file_load` for data that can be reloaded from a recorded file source
     and replayed through an ordered sequence of :class:`ReplayStep` objects.

   - Use :func:`script` for console-derived and other multi-input results. Script
     specs may reference any number of :class:`ScriptInput` records. Each input stores
     an immutable replay name, a historical display label, optional live manager node
     identity and role-specific ``node_snapshot_token``, whether it represents source
     or displayed data, and an optional nested provenance snapshot.

Each transformation is represented by an immutable
:class:`ToolProvenanceOperation` subclass whose serialized fields are safe to persist
in JSON. Persisted specs are the source of truth for workspace save/load. Runtime
reload, copied code, and shared-input deduplication compile those specs on demand
through :mod:`erlab.interactive.imagetool._provenance._graph`; the graph itself is not
saved.

Operations are intentionally primitive: a user action that needs several public API
calls should normally remain several structured operations rather than falling back to
one raw script block. When those primitive operations should behave as one edit unit,
authoring code can stamp them with optional :class:`OperationGroupMarker` metadata.
Group markers are edit/copy metadata only. They do not change replay semantics, and
broken or partial groups can be stripped without changing the numerical operation list.

Manager children opened from an ImageTool cursor or bin selection keep the explicit
``qsel`` or ``isel`` arguments generated when the child is opened. Legacy selection
source bindings are materialized once into a normal source spec before refresh.

Adding a new provenance-carrying operation follows the same pattern every time:

1. Define a new :class:`ToolProvenanceOperation` subclass with a unique ``op``
   discriminator literal. Subclasses register themselves automatically so serialized
   payloads can dispatch to the right model.

2. Prefer the annotated provenance field aliases defined in this module for hashable
   dimension identifiers and dim-keyed mappings so runtime values stay decoded while
   JSON dumps remain lossless.

3. If the operation needs already-encoded payloads such as xarray objects or array-like
   vertex data, validate them with
   :meth:`ToolProvenanceOperation._validate_encoded_field` and expose decoded
   convenience properties for runtime use.

4. Implement :meth:`ToolProvenanceOperation.apply` so it transforms a derived array
   using only its recorded parameters. Operations are unary so their meaning remains
   stable when users reorder them.

5. Implement :meth:`ToolProvenanceOperation.expression_code` and
   :meth:`ToolProvenanceOperation.derivation_label` so copied code and manager
   derivation entries come from the same operation metadata. Override
   :meth:`ToolProvenanceOperation.derivation_entry` only when the operation is not a
   structured transform, such as free-form script code.

6. If the public API for the operation is mutating and cannot be represented as a
   single expression, implement :meth:`ToolProvenanceOperation.statement_code` and set
   :attr:`ToolProvenanceOperation.statement_mutates_input`. Mixed expression and
   statement operation chains are still emitted as ordinary public Python code.

7. If several primitive operations form one user-editable action, stamp the complete
   contiguous sequence with :func:`stamp_operation_group`. Group-aware dialogs should
   validate the complete group with :func:`operation_group_range` plus any
   operation-specific rule, and clipboard code should call
   :func:`strip_partial_operation_groups` when users copy a subset of rows. Paste code
   should call :func:`restamp_operation_groups` before appending copied rows to another
   provenance chain.

8. If the operation maps cleanly from manager-console calls, declare
   :attr:`ToolProvenanceOperation.console_patterns` or implement
   :meth:`ToolProvenanceOperation.from_console_call`. Unsupported or ambiguous console
   calls should return ``None`` so they remain valid script provenance instead of being
   recorded as lossy structured operations.

9. Make generated code executable in the replay namespace. Use the literal helpers in
   this module for persisted values, and make ``expression_code`` honor the input name
   it is given. The base ``replay_code`` wrapper assigns the caller-selected output
   name.

10. Export the operation class from the concrete operation catalog so runtime call
    sites can instantiate it directly.

11. Add tests that cover round-trip validation, :meth:`apply`, derivation text/code,
    console matching when supported, and any save/load or reload path that persists the
    new operation. For grouped operations, also cover full-group copy/paste, partial
    group stripping, group replacement/deletion, and generated-code execution.

Parsing of serialized payloads happens only through :func:`parse_tool_provenance_spec`
and :func:`parse_tool_provenance_operation`. Runtime authoring code should create specs
with :func:`full_data`, :func:`public_data`, :func:`selection`, :func:`file_load`, or
:func:`script`, then instantiate operation models from
:mod:`erlab.interactive.imagetool._provenance._operations` directly.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import importlib
import inspect
import keyword
import typing
import uuid
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance._code import (
    _DATAARRAY_MARKER,
    _DATASET_MARKER,
    _FIT_DATASET_MARKER,
    _MAPPING_MARKER,
    _SLICE_MARKER,
    _TUPLE_MARKER,
    _dynamic_nonuniform_restore_replay_code,
    _expression_receiver_code,
    _migrate_legacy_nonuniform_restore_code,
    _receiver_path,
    _script_codes_output_name,
    _simplify_display_code,
    _validate_active_name,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

_SourceKind: typing.TypeAlias = typing.Literal["full_data", "public_data", "selection"]
_ReplayInputPolicy: typing.TypeAlias = typing.Literal["current", "restored"]
ScriptInputDataRole: typing.TypeAlias = typing.Literal["source", "displayed"]
FileLoadSourceStatus: typing.TypeAlias = typing.Literal[
    "loadable",
    "no-file-load-source",
    "missing-file",
    "no-replay-call",
    "missing-loader",
]

_DEFAULT_REPLAY_SEED_CODE = "derived = data"
_PROMOTED_1D_SOURCE_ATTR = "_erlab_promoted_from_1d_source"
_SORT_COORD_ORDER_DERIVATION_LABEL = "Sort coordinates to parent order"


@dataclass(frozen=True)
class DerivationEntry:
    """One user-visible step in a provenance derivation listing."""

    label: str
    code: str | None
    copyable: bool = False


@dataclass(frozen=True)
class _ProvenanceStepRef:
    """Location of one replayable provenance row inside a spec."""

    kind: typing.Literal["start", "file_load", "operation", "script_input"]
    operation_index: int | None = None
    script_input_index: int | None = None


@dataclass(frozen=True)
class _ProvenanceDisplayRow:
    """One displayed provenance row with optional edit/replay references."""

    entry: DerivationEntry
    edit_ref: _ProvenanceStepRef | None = None
    replay_ref: _ProvenanceStepRef | None = None
    scope: typing.Literal["display", "source"] = "display"
    children: tuple[_ProvenanceDisplayRow, ...] = ()
    script_input_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ProvenanceReorderBlockRef:
    """Original operation range represented by one reorder-dialog row."""

    start: int
    stop: int


@dataclass(frozen=True)
class _ProvenanceReorderBlock:
    """One atomic provenance block exposed by the reorder dialog."""

    ref: _ProvenanceReorderBlockRef
    entries: tuple[DerivationEntry, ...]
    label: str | None = None
    tooltip: str | None = None


@dataclass(frozen=True)
class _ProvenanceReorderSectionRef:
    """Contiguous operation range that may be permuted."""

    start: int
    stop: int


@dataclass(frozen=True)
class _ProvenanceReorderSection:
    """Movable operation blocks bounded by fixed provenance semantics."""

    ref: _ProvenanceReorderSectionRef
    label: str
    blocks: tuple[_ProvenanceReorderBlock, ...]


@dataclass(frozen=True)
class _ProvenanceDisplayContext:
    """Cheap metadata known while streamlining provenance display rows."""

    dims: tuple[Hashable, ...] | None = None
    sizes: dict[Hashable, int] | None = None

    @classmethod
    def from_source(
        cls,
        source_kind: _SourceKind,
        parent_data: xr.DataArray | None,
    ) -> _ProvenanceDisplayContext:
        if parent_data is None:
            return cls()
        if source_kind != "full_data" and cls.dims_may_restore_nonuniform(
            parent_data.dims
        ):
            return cls()
        return cls(tuple(parent_data.dims), dict(parent_data.sizes))

    @staticmethod
    def dims_may_restore_nonuniform(dims: Sequence[Hashable]) -> bool:
        return any(isinstance(dim, str) and dim.endswith("_idx") for dim in dims)

    @staticmethod
    def _isel_indexer_size(indexer: typing.Any, size: int) -> tuple[bool, int | None]:
        if isinstance(indexer, slice):
            return True, len(range(*indexer.indices(size)))
        if isinstance(indexer, (bool, np.bool_)):
            return False, None
        if isinstance(indexer, (int, np.integer)):
            return True, None
        if isinstance(indexer, xr.DataArray):
            return False, None
        if isinstance(indexer, np.ndarray):
            if indexer.ndim == 0:
                return True, None
            if indexer.ndim != 1 or indexer.dtype == np.dtype(bool):
                return False, None
            return True, int(indexer.shape[0])
        if isinstance(indexer, range):
            return True, len(indexer)
        if isinstance(indexer, Sequence) and not isinstance(indexer, (bytes, str)):
            if len(indexer) and all(
                isinstance(item, (bool, np.bool_)) for item in indexer
            ):
                return False, None
            return True, len(indexer)
        return False, None

    def _with_dims(self, dims: Iterable[Hashable]) -> _ProvenanceDisplayContext:
        dims = tuple(dims)
        return type(self)(
            dims,
            None
            if self.sizes is None
            else {dim: self.sizes[dim] for dim in dims if dim in self.sizes},
        )

    def advance(self, operation: ToolProvenanceOperation) -> _ProvenanceDisplayContext:
        operation_name = getattr(operation, "op", None)
        if operation_name == "rename":
            return self
        if operation_name == "source_view":
            if self.dims is not None and (
                getattr(operation, "source_kind", None) == "full_data"
                or not self.dims_may_restore_nonuniform(self.dims)
            ):
                return self
            return type(self)()
        if operation_name in {"qsel", "sel"}:
            return (
                self if not getattr(operation, "decoded_kwargs", {}) else type(self)()
            )
        if operation_name == "isel":
            kwargs = getattr(operation, "decoded_kwargs", {})
            if not kwargs:
                return self
            if self.dims is None or self.sizes is None:
                return type(self)()
            dims = list(self.dims)
            sizes = dict(self.sizes)
            for dim, indexer in kwargs.items():
                if dim not in dims or dim not in sizes:
                    return type(self)()
                known, indexer_size = self._isel_indexer_size(indexer, sizes[dim])
                if not known:
                    return type(self)()
                if indexer_size is None:
                    dims.remove(dim)
                    sizes.pop(dim, None)
                else:
                    sizes[dim] = indexer_size
            return type(self)(
                tuple(dims),
                {dim: sizes[dim] for dim in dims if dim in sizes},
            )
        if operation_name == "transpose":
            if self.dims is None:
                return type(self)()
            operation_dims = getattr(operation, "dims", None)
            target_dims = (
                tuple(operation_dims)
                if operation_dims is not None
                else tuple(reversed(self.dims))
            )
            if set(target_dims) != set(self.dims) or len(target_dims) != len(self.dims):
                return type(self)()
            return self._with_dims(target_dims)
        if operation_name == "squeeze":
            if self.dims is None or self.sizes is None:
                return type(self)()
            operation_dims = getattr(operation, "dims", None)
            if operation_dims is None:
                return self._with_dims(
                    [dim for dim in self.dims if self.sizes.get(dim) != 1]
                )
            target_dims = tuple(operation_dims)
            if any(
                dim not in self.dims or dim not in self.sizes for dim in target_dims
            ):
                return type(self)()
            singleton_dims = [dim for dim in target_dims if self.sizes.get(dim) == 1]
            if not singleton_dims:
                return self
            if len(singleton_dims) != len(target_dims):
                return type(self)()
            return self._with_dims(
                dim for dim in self.dims if dim not in singleton_dims
            )
        if operation_name == "restore_nonuniform_dims":
            if self.dims is not None and not self.dims_may_restore_nonuniform(
                self.dims
            ):
                return self
            return type(self)()
        return type(self)()


@dataclass(frozen=True)
class ScriptInputDependencyRef:
    """Live manager dependency captured by a script input."""

    name: str
    label: str
    node_uid: str
    node_snapshot_token: str | None = None
    data_role: ScriptInputDataRole = "displayed"


class OperationGroupMarker(pydantic.BaseModel):
    """Optional edit/copy metadata shared by a contiguous operation group."""

    kind: str
    id: str
    index: int
    size: int
    focus: str | None = pydantic.Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def _validate_marker(self) -> typing.Self:
        if not self.kind:
            raise ValueError("operation group kind must not be empty")
        if not self.id:
            raise ValueError("operation group id must not be empty")
        if self.index < 0:
            raise ValueError("operation group index must be non-negative")
        if self.size <= 0:
            raise ValueError("operation group size must be positive")
        if self.index >= self.size:
            raise ValueError("operation group index must be smaller than size")
        return self


def _encode_fit_dataset(value: xr.Dataset) -> str:
    return base64.b64encode(
        erlab.interactive.utils._serialize_fit_dataset_blob(value).tobytes()
    ).decode("ascii")


def _decode_fit_dataset(value: typing.Any) -> xr.Dataset:
    payload = (
        np.frombuffer(base64.b64decode(value.encode("ascii")), dtype=np.uint8)
        if isinstance(value, str)
        else value
    )
    return erlab.interactive.utils._deserialize_fit_dataset_blob(payload)


def _is_whole_array_rename_entry(entry: DerivationEntry) -> bool:
    code = entry.code
    if code is None:
        return False
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    match module.body:
        case [
            ast.Assign(
                targets=[ast.Name(id=target_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=receiver_name, ctx=ast.Load()),
                        attr="rename",
                    ),
                    args=args,
                    keywords=keywords,
                ),
            )
        ] if target_name == receiver_name:
            pass
        case _:
            return False

    if len(args) == 1 and not keywords:
        return isinstance(args[0], ast.Constant) and isinstance(
            args[0].value, (str, type(None))
        )
    if args or len(keywords) != 1:
        return False
    keyword = keywords[0]
    return (
        keyword.arg == "new_name_or_name_dict"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, (str, type(None)))
    )


def encode_provenance_value(value: typing.Any) -> typing.Any:
    """Encode non-JSON provenance values into a JSON-safe representation."""
    if isinstance(value, xr.Dataset):
        if any(str(var).endswith("modelfit_results") for var in value.data_vars):
            return {_FIT_DATASET_MARKER: _encode_fit_dataset(value)}
        return {_DATASET_MARKER: value.to_dict(data="list")}
    if isinstance(value, xr.DataArray):
        return {_DATAARRAY_MARKER: value.to_dict(data="list")}
    value = erlab.utils.misc._convert_to_native(value)
    if isinstance(value, slice):
        return {
            _SLICE_MARKER: [
                encode_provenance_value(value.start),
                encode_provenance_value(value.stop),
                encode_provenance_value(value.step),
            ]
        }
    if isinstance(value, np.ndarray):
        return [encode_provenance_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [encode_provenance_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_provenance_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            _MAPPING_MARKER: [
                [_encode_provenance_hashable(key), encode_provenance_value(item)]
                for key, item in value.items()
            ]
        }
    return value


def decode_provenance_value(value: typing.Any) -> typing.Any:
    """Decode values produced by :func:`encode_provenance_value`."""
    if isinstance(value, Mapping):
        if _FIT_DATASET_MARKER in value:
            return _decode_fit_dataset(value[_FIT_DATASET_MARKER])
        if _DATASET_MARKER in value:
            return xr.Dataset.from_dict(
                typing.cast("dict[str, typing.Any]", value[_DATASET_MARKER])
            )
        if _DATAARRAY_MARKER in value:
            return xr.DataArray.from_dict(
                typing.cast("dict[str, typing.Any]", value[_DATAARRAY_MARKER])
            )
        if _SLICE_MARKER in value:
            start, stop, step = typing.cast("list[typing.Any]", value[_SLICE_MARKER])
            return slice(
                decode_provenance_value(start),
                decode_provenance_value(stop),
                decode_provenance_value(step),
            )
        if _TUPLE_MARKER in value:
            return tuple(
                decode_provenance_value(item)
                for item in typing.cast("list[typing.Any]", value[_TUPLE_MARKER])
            )
        if _MAPPING_MARKER in value:
            return {
                decode_provenance_value(key): decode_provenance_value(item)
                for key, item in typing.cast(
                    "list[list[typing.Any]]", value[_MAPPING_MARKER]
                )
            }
        return {
            decode_provenance_value(key): decode_provenance_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [decode_provenance_value(item) for item in value]
    return erlab.utils.misc._convert_to_native(value)


def _normalize_provenance_hashable(value: typing.Any) -> Hashable:
    value = erlab.utils.misc._convert_to_native(value)
    if isinstance(value, tuple):
        return tuple(_normalize_provenance_hashable(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return typing.cast("Hashable", value)
    raise TypeError(
        "provenance hashable fields only support str, int, float, bool, None, "
        "and tuples composed of those values"
    )


def _encode_provenance_hashable(value: typing.Any) -> typing.Any:
    normalized = _normalize_provenance_hashable(value)
    if isinstance(normalized, tuple):
        return {
            _TUPLE_MARKER: [_encode_provenance_hashable(item) for item in normalized]
        }
    return normalized


def _is_identifier_string_mapping(value: Mapping[typing.Any, typing.Any]) -> bool:
    return all(
        isinstance(key, str) and key.isidentifier() and not keyword.iskeyword(key)
        for key in value
    )


def _format_derivation_value(value: typing.Any) -> str:
    decoded = decode_provenance_value(value)
    if isinstance(decoded, Mapping):
        return erlab.interactive.utils.format_kwargs(dict(decoded))
    if isinstance(decoded, (tuple, list)):
        return repr(tuple(decoded))
    return repr(decoded)


def _ensure_float_tuple(
    value: Sequence[typing.Any], *, expected_len: int | None = None
) -> tuple[float, ...]:
    items = tuple(float(item) for item in value)
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)}")
    return items


def _coerce_float_sequence(value: typing.Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("expected an array-like sequence")
    return [float(item) for item in value]


def _starting_data_for_source_kind(
    source_kind: _SourceKind,
    parent_data: xr.DataArray,
) -> xr.DataArray:
    if source_kind == "full_data":
        return parent_data.copy(deep=False)
    return erlab.utils.array._restore_nonuniform_dims(parent_data.copy(deep=False))


_OPERATION_TYPES: dict[str, type[ToolProvenanceOperation]] = {}


def _ensure_operation_catalog_loaded() -> None:
    """Load concrete operations before resolving a serialized discriminator."""
    importlib.import_module("erlab.interactive.imagetool._provenance._operations")


def _operation_is(operation: ToolProvenanceOperation, *operation_names: str) -> bool:
    return getattr(operation, "op", None) in operation_names


def _operation_type(op: str) -> type[ToolProvenanceOperation]:
    _ensure_operation_catalog_loaded()
    operation_type = _OPERATION_TYPES.get(op)
    if operation_type is None:
        raise ValueError(
            f"Unknown provenance operation {op!r}; "
            f"expected one of {sorted(_OPERATION_TYPES)}"
        )
    return operation_type


def _operation_instance(op: str, **kwargs: typing.Any) -> ToolProvenanceOperation:
    return _operation_type(op)(**kwargs)


_UNPARSED_LITERAL = object()


def _literal_node_value(node: ast.AST) -> typing.Any:
    try:
        return ast.literal_eval(node)
    except (SyntaxError, TypeError, ValueError):
        pass

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice"
        and len(node.args) <= 3
        and not node.keywords
    ):
        values = [_literal_node_value(arg) for arg in node.args]
        if any(value is _UNPARSED_LITERAL for value in values):
            return _UNPARSED_LITERAL
        return slice(*values)

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "array"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"np", "numpy"}
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _literal_node_value(node.args[0])
        if value is _UNPARSED_LITERAL:
            return _UNPARSED_LITERAL
        return np.asarray(value)

    return _UNPARSED_LITERAL


def _literal_call_args_kwargs(
    call: ast.Call,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]] | None:
    args = tuple(_literal_node_value(arg) for arg in call.args)
    if any(value is _UNPARSED_LITERAL for value in args):
        return None
    kwargs: dict[str, typing.Any] = {}
    for call_keyword in call.keywords:
        if call_keyword.arg is None:
            return None
        value = _literal_node_value(call_keyword.value)
        if value is _UNPARSED_LITERAL:
            return None
        kwargs[call_keyword.arg] = value
    return args, kwargs


def _operation_from_self_assignment_call(
    target_name: str,
    call: ast.Call,
) -> ToolProvenanceOperation | None:
    parsed_args = _literal_call_args_kwargs(call)
    if parsed_args is None:
        return None
    args, kwargs = parsed_args

    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"mean", "min", "max", "sum"}
        and not args
        and not kwargs
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Attribute)
        and call.func.value.func.attr == "coarsen"
        and isinstance(call.func.value.func.value, ast.Name)
        and call.func.value.func.value.id == target_name
    ):
        parsed_coarsen_args = _literal_call_args_kwargs(call.func.value)
        if parsed_coarsen_args is None:
            return None
        coarsen_args, coarsen_kwargs = parsed_coarsen_args
        return operation_from_console_call(
            ConsoleCall(
                dataarray_method="coarsen",
                args=coarsen_args,
                kwargs={**coarsen_kwargs, "_reducer": call.func.attr},
                display_code=ast.unparse(call),
                has_extra_tracked_inputs=False,
            )
        )

    if not isinstance(call.func, ast.Attribute):
        return None
    receiver = _receiver_path(call.func.value)
    if receiver is None:
        return None
    receiver_name, accessor_path = receiver
    if receiver_name != target_name:
        return None

    operation = operation_from_console_call(
        ConsoleCall(
            dataarray_method=call.func.attr if not accessor_path else None,
            accessor_path=(*accessor_path, call.func.attr) if accessor_path else (),
            args=args,
            kwargs=kwargs,
            display_code=ast.unparse(call),
            has_extra_tracked_inputs=False,
        )
    )
    if operation is not None:
        return operation
    if accessor_path:
        return None
    return operation_from_console_call(
        ConsoleCall(
            accessor_path=(call.func.attr,),
            args=args,
            kwargs=kwargs,
            display_code=ast.unparse(call),
            has_extra_tracked_inputs=False,
        )
    )


def _structured_operation_from_script_code(
    operation: ToolProvenanceOperation,
    *,
    current_name: str | None,
) -> ToolProvenanceOperation:
    if current_name is None:
        return operation
    code = getattr(operation, "code", None)
    if not getattr(operation, "copyable", False) or code is None:
        return operation
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return operation
    if (
        len(module.body) != 1
        or not isinstance(module.body[0], ast.Assign)
        or len(module.body[0].targets) != 1
        or not isinstance(module.body[0].targets[0], ast.Name)
        or not isinstance(module.body[0].value, ast.Call)
    ):
        return operation
    target_name = module.body[0].targets[0].id
    if target_name != current_name:
        return operation
    structured = _operation_from_self_assignment_call(
        current_name, module.body[0].value
    )
    return operation if structured is None else structured


def _callable_paths(func: Callable[..., typing.Any]) -> set[str]:
    paths: set[str] = set()
    for value in (func, inspect.unwrap(func)):
        module = getattr(value, "__module__", None)
        if not isinstance(module, str):
            continue
        for attr in ("__qualname__", "__name__"):
            name = getattr(value, attr, None)
            if isinstance(name, str):
                paths.add(f"{module}.{name}")
    return paths


class ConsoleCall:
    """Normalized runtime call observed by the ImageTool manager console.

    Console proxies create this descriptor after unwrapping provenance-aware tool
    handles to raw :class:`xarray.DataArray` arguments. Operation classes inspect the
    invoked callable, DataArray method/accessor context, and raw arguments to decide
    whether the call can be represented as structured provenance.
    """

    __slots__ = (
        "accessor_path",
        "args",
        "dataarray_method",
        "display_code",
        "func",
        "has_extra_tracked_inputs",
        "kwargs",
        "receiver_data",
    )

    def __init__(
        self,
        *,
        func: Callable[..., typing.Any] | None = None,
        dataarray_method: str | None = None,
        accessor_path: Sequence[str] = (),
        args: Sequence[typing.Any] = (),
        kwargs: Mapping[str, typing.Any] | None = None,
        display_code: str,
        has_extra_tracked_inputs: bool,
        receiver_data: xr.DataArray | None = None,
    ) -> None:
        self.func = func
        self.dataarray_method = dataarray_method
        self.accessor_path = tuple(accessor_path)
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.display_code = display_code
        self.has_extra_tracked_inputs = has_extra_tracked_inputs
        self.receiver_data = receiver_data


class ConsoleOperationPattern:
    """Declarative matcher for direct console-to-operation mappings.

    Use this for calls where public arguments map directly onto operation model
    fields. Store module-function targets as strings so importing this module does not
    resolve lazy ERLab analysis modules; the matcher compares those strings with the
    callable that the console actually invoked. More complex calls should implement
    :meth:`ToolProvenanceOperation.from_console_call`.
    """

    __slots__ = (
        "accessor_path",
        "dataarray_method",
        "defaults",
        "field_aliases",
        "fields",
        "ignored_defaults",
        "kwargs_field",
        "mapping_kwargs",
        "targets",
    )

    def __init__(
        self,
        *,
        target: Callable[..., typing.Any] | str | None = None,
        dataarray_method: str | None = None,
        accessor_path: Sequence[str] = (),
        fields: Sequence[str] = (),
        field_aliases: Mapping[str, str] | None = None,
        kwargs_field: str | None = None,
        mapping_kwarg: str | Sequence[str] | None = None,
        defaults: Mapping[str, typing.Any] | None = None,
        ignored_defaults: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self.targets = () if target is None else (target,)
        self.dataarray_method = dataarray_method
        self.accessor_path = tuple(accessor_path)
        self.fields = tuple(fields)
        self.field_aliases = dict(field_aliases or {})
        self.kwargs_field = kwargs_field
        mapping_kwargs: tuple[str, ...]
        if isinstance(mapping_kwarg, str):
            mapping_kwargs = (mapping_kwarg,)
        else:
            mapping_kwargs = tuple(mapping_kwarg or ())
        self.mapping_kwargs = mapping_kwargs
        self.defaults = dict(defaults or {})
        self.ignored_defaults = dict(ignored_defaults or {})

    def match(self, call: ConsoleCall) -> dict[str, typing.Any] | None:
        if call.has_extra_tracked_inputs:
            return None
        if self.targets:
            if call.func is None:
                return None
            call_paths = _callable_paths(call.func)
            if not any(
                target in call_paths
                if isinstance(target, str)
                else inspect.unwrap(call.func) is inspect.unwrap(target)
                for target in self.targets
            ):
                return None
        if self.dataarray_method is not None:
            if call.dataarray_method != self.dataarray_method:
                return None
        elif call.dataarray_method is not None:
            return None
        if self.accessor_path != call.accessor_path:
            return None

        kwargs = dict(call.kwargs)
        for public_name, field_name in self.field_aliases.items():
            if public_name not in kwargs:
                continue
            if field_name in kwargs:
                return None
            kwargs[field_name] = kwargs.pop(public_name)
        for key, default in self.ignored_defaults.items():
            if key not in kwargs:
                continue
            if not _console_values_equal(kwargs[key], default):
                return None
            kwargs.pop(key)

        if self.kwargs_field is not None:
            mapped = _console_mapping_values(
                call.args, kwargs, mapping_kwargs=self.mapping_kwargs
            )
            if mapped is None:
                return None
            return {self.kwargs_field: mapped}

        if len(call.args) > len(self.fields):
            return None
        values: dict[str, typing.Any] = dict(zip(self.fields, call.args, strict=False))
        for field in self.fields[len(call.args) :]:
            if field in kwargs:
                values[field] = kwargs.pop(field)
            elif field in self.defaults:
                values[field] = self.defaults[field]
            else:
                return None
        for key, value in kwargs.items():
            if key not in self.defaults:
                return None
            values[key] = value
        for key, value in self.defaults.items():
            values.setdefault(key, value)
        return values


def _console_values_equal(value: typing.Any, default: typing.Any) -> bool:
    if (
        isinstance(value, float)
        and isinstance(default, float)
        and np.isnan(value)
        and np.isnan(default)
    ):
        return True
    return value == default


def _console_mapping_values(
    args: Sequence[typing.Any],
    kwargs: Mapping[str, typing.Any],
    *,
    mapping_kwargs: Sequence[str] = (),
) -> dict[typing.Any, typing.Any] | None:
    if len(args) > 1:
        return None
    mapped: dict[typing.Any, typing.Any] = {}
    remaining = dict(kwargs)
    if args:
        if args[0] is None:
            pass
        elif not isinstance(args[0], Mapping):
            return None
        else:
            mapped.update(args[0])
    for mapping_kwarg in mapping_kwargs:
        if mapping_kwarg not in remaining:
            continue
        value = remaining.pop(mapping_kwarg)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            return None
        mapped.update(value)
    mapped.update(remaining)
    return mapped


def _assignment_code(
    output_name: str,
    expression: str,
    *,
    line_length: int = 88,
) -> str:
    """Assign an expression while preserving readable call formatting."""
    code = f"{output_name} = {expression}"
    if "\n" in expression or len(code) <= line_length:
        return code
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return code
    if not isinstance(parsed.body, ast.Call):
        return code

    function_code = ast.get_source_segment(expression, parsed.body.func)
    if function_code is None:
        return code
    arguments: list[str] = []
    for argument in parsed.body.args:
        argument_code = ast.get_source_segment(expression, argument)
        if argument_code is None:
            return code
        if isinstance(argument, ast.GeneratorExp):
            argument_code = f"({argument_code})"
        arguments.append(argument_code)
    for keyword_arg in parsed.body.keywords:
        value_code = ast.get_source_segment(expression, keyword_arg.value)
        if value_code is None:
            return code
        prefix = "**" if keyword_arg.arg is None else f"{keyword_arg.arg}="
        arguments.append(f"{prefix}{value_code}")
    if not arguments:
        return code
    return "\n".join(
        (
            f"{output_name} = {function_code}(",
            *(f"    {argument}," for argument in arguments),
            ")",
        )
    )


class ToolProvenanceOperation(pydantic.BaseModel):
    """Base class for typed operations stored in :class:`ToolProvenanceSpec`.

    New operations should keep runtime fields in their decoded Python form, prefer the
    annotated provenance field aliases in this module for lossless JSON serialization,
    implement :meth:`apply` to replay the transformation, implement
    :meth:`expression_code` to emit the public Python expression for the same
    transformation, and implement :meth:`derivation_label` to describe the step in
    manager UI.

    Operation instances store the exact arguments used by live refresh, replay graph
    execution, copied code, and derivation display. If a public console call maps
    exactly to an operation, expose it with ``console_patterns`` or
    :meth:`from_console_call`; ambiguous calls should return ``None`` so script
    provenance records the original code instead.
    """

    live_applicable: typing.ClassVar[bool] = True
    batch_available: typing.ClassVar[bool] = False
    console_applies_to_receiver: typing.ClassVar[bool] = False
    statement_mutates_input: typing.ClassVar[bool] = False
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = ()
    group: OperationGroupMarker | None = pydantic.Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()

        if "op" not in cls.__dict__.get("__annotations__", {}):
            return

        op_field = cls.model_fields.get("op")
        op = None if op_field is None else op_field.default
        if not isinstance(op, str):
            raise TypeError(f"{cls.__name__} must define `op` with a string default")

        existing = _OPERATION_TYPES.get(op)
        if existing is not None and existing is not cls:
            raise ValueError(f"Duplicate provenance operation discriminator {op!r}")

        _OPERATION_TYPES[op] = cls

    @staticmethod
    def _json_encode_field(value: typing.Any) -> typing.Any:
        return encode_provenance_value(value)

    @classmethod
    def _validate_encoded_field(
        cls,
        value: typing.Any,
        *,
        error: str,
        predicate: Callable[[typing.Any], bool],
    ) -> typing.Any:
        encoded = encode_provenance_value(value)
        if not predicate(encoded):
            raise TypeError(error)
        return encoded

    @staticmethod
    def _decode_stored_field(value: typing.Any) -> typing.Any:
        return decode_provenance_value(value)

    @staticmethod
    def _coerce_hashable_field(value: typing.Any) -> Hashable:
        return _normalize_provenance_hashable(decode_provenance_value(value))

    @staticmethod
    def _coerce_hashable_tuple_field(
        value: typing.Any, *, expected_len: int | None = None
    ) -> tuple[Hashable, ...]:
        decoded = decode_provenance_value(value)
        if not isinstance(decoded, Sequence) or isinstance(decoded, str):
            raise TypeError("expected a sequence of provenance-compatible hashables")
        items = tuple(_normalize_provenance_hashable(item) for item in decoded)
        if expected_len is not None and len(items) != expected_len:
            raise ValueError(f"Expected {expected_len} items, got {len(items)}")
        return items

    @staticmethod
    def _coerce_hashable_mapping_field(
        value: typing.Any,
        *,
        value_coerce: Callable[[typing.Any], typing.Any] | None = None,
    ) -> dict[Hashable, typing.Any]:
        if value is None:
            return {}
        decoded = decode_provenance_value(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("expected a mapping with provenance-compatible keys")
        transform = (lambda item: item) if value_coerce is None else value_coerce
        return {
            _normalize_provenance_hashable(key): transform(item)
            for key, item in decoded.items()
        }

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        """Apply this operation to the current derived array.

        Subclasses that participate in live refresh or executable provenance replay
        should reimplement this method. The implementation must be deterministic for
        the operation's stored model fields and must not mutate ``data`` in place.

        Parameters
        ----------
        data
            Array produced by the preceding replay step.

        Returns
        -------
        xarray.DataArray
            Array after this operation has been applied.

        Notes
        -----
        Operations that only emit generated code should set ``live_applicable = False``
        and raise from this method.

        .. versionchanged:: 3.25.0
           Operations receive only the current derived array. Record every input needed
           for deterministic replay in the operation model.
        """
        raise NotImplementedError

    def _apply_schema_v2(
        self,
        data: xr.DataArray,
        *,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        """Apply an operation deserialized from a schema-v2 parent-data context.

        All current operations are unary and the default ignores the legacy context.
        Third-party operations saved under schema v2 may have overridden this method,
        so migrated steps retain the original ``parent_data`` call contract. Remove
        this method, ``ReplayStep.legacy_context``, and the schema-v2 migration
        together when old saved workspace support is retired.
        """
        return self.apply(data)

    def derivation_entry(self) -> DerivationEntry:
        """Return the user-visible derivation entry for this operation.

        Structured transform subclasses should normally reimplement
        :meth:`derivation_label` and :meth:`expression_code` instead of overriding this
        method. Override this method only for operations whose display entry cannot be
        represented as a label plus parameterized operation code, such as stored
        free-form script code.

        Returns
        -------
        DerivationEntry
            Label, replay code, and copyability flag shown in derivation UI and used by
            legacy derivation-code paths.

        Notes
        -----
        The base implementation keeps the legacy ``derived`` replay contract while
        letting concrete operations emit expression code for any input variable.

        Use ``DerivationEntry(..., code=None)`` instead when the step should remain
        visible in the derivation list but code generation should stop and return
        ``None``.
        """
        return DerivationEntry(
            self.derivation_label(),
            self.replay_code("derived", output_name="derived", source_name="data"),
            True,
        )

    def derivation_label(self) -> str:
        """Return the derivation-list label for this operation.

        Subclasses should reimplement this for every structured operation that uses
        the base :meth:`derivation_entry` implementation.

        Returns
        -------
        str
            Human-readable operation label for derivation/history UI. Include the
            operation's meaningful parameters when they help explain the produced
            data.
        """
        raise NotImplementedError

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        """Return a Python expression applying this operation to an input name.

        Subclasses should reimplement this for every structured operation that can be
        represented as ordinary user-facing Python code. The emitted expression should
        be complete, public-API-based, and free of assignment to hardcoded temporary
        names.

        Parameters
        ----------
        input_name
            Python expression or identifier for the array produced by the previous
            replay step. Use this exact value as the operation receiver/input instead
            of assuming a variable name such as ``derived``.
        source_name
            Python expression or identifier for the original public input array for
            the enclosing replay sequence. Operations that need parent/source context,
            such as coordinate-order restoration, should use this name for that
            context. Operations that only transform ``input_name`` may ignore it.

        Returns
        -------
        str
            Python expression that evaluates to the transformed DataArray.
        """
        raise NotImplementedError

    def statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
    ) -> str:
        """Return Python statements applying this operation to an input name.

        Most structured operations are expression-like and should implement
        :meth:`expression_code`. Operations whose public API is mutating, such as
        accessor setters, should implement this method instead and assign their result
        to ``output_name``.
        """
        raise NotImplementedError

    def _statement_replay_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        """Call statement replay with internal graph-emission context."""
        return self.statement_code(
            input_name,
            output_name=output_name,
            source_name=source_name,
        )

    def replay_code(
        self,
        input_name: str,
        *,
        output_name: str | None = None,
        source_name: str | None = None,
        reserved_names: Collection[str] = (),
    ) -> str:
        """Return replay code for this operation with caller-selected names.

        This method usually should not be reimplemented by subclasses. Implement
        :meth:`expression_code` instead so replay graph emission, dialog copy code, and
        derivation entries all share the same operation expression.

        Parameters
        ----------
        input_name
            Python expression or identifier for the array produced by the previous
            replay step.
        output_name
            Variable name to assign the transformed value to. If ``None``, return only
            the expression from :meth:`expression_code`.
        source_name
            Python expression or identifier for the original public input array for
            the enclosing replay sequence. Passed through to :meth:`expression_code`.
        reserved_names
            Names already used by the enclosing replay sequence. Statement-based
            operations use these to choose collision-free auxiliary targets.

        Returns
        -------
        str
            Either an assignment statement when ``output_name`` is provided, or a bare
            expression when it is ``None``.
        """
        try:
            expression = self.expression_code(input_name, source_name=source_name)
        except NotImplementedError:
            if output_name is None:
                raise
            return self._statement_replay_code(
                input_name,
                output_name=output_name,
                source_name=source_name,
                reserved_names=reserved_names,
            )
        if output_name is None:
            return expression
        return _assignment_code(output_name, expression)

    def preferred_replay_output_name(self) -> str | None:
        """Return a semantic output name when this operation changes value kind.

        Most transformations preserve the meaning of their input name and return
        ``None``. Operations that produce a categorically different result may provide
        a name which the replay graph uses when it matches the enclosing script's
        active output.
        """
        return None

    def preferred_replay_input_name(self) -> str | None:
        """Return a semantic name for an input reused by generated code.

        Most operations use their input once and should return ``None``. Operations
        that refer to the same intermediate several times may provide a descriptive
        name so replay code does not expose an internal graph temporary.
        """
        return None

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        """Build an operation from a normalized manager-console call.

        Subclasses should reimplement this when a console call maps cleanly to the
        operation but cannot be expressed with declarative ``console_patterns`` alone.
        Return ``None`` for unsupported, ambiguous, or lossy calls so the original
        console code remains recorded as script provenance.

        Parameters
        ----------
        call
            Normalized descriptor for the function, method, or accessor call observed
            by the ImageTool manager console, including unwrapped arguments, keyword
            arguments, display code, and receiver data when available.

        Returns
        -------
        ToolProvenanceOperation or None
            Operation instance when the call is exactly representable by this
            operation class; otherwise ``None``.
        """
        for pattern in cls.console_patterns:
            values = pattern.match(call)
            if values is None:
                continue
            with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
                return cls(**values)
        return None


def _default_seed_code_for_operations(
    operations: Sequence[ToolProvenanceOperation],
) -> str:
    if any(operation.statement_mutates_input for operation in operations):
        return "derived = data.copy(deep=False)"
    return _DEFAULT_REPLAY_SEED_CODE


def _operation_without_group(
    operation: ToolProvenanceOperation,
) -> ToolProvenanceOperation:
    if operation.group is None:
        return operation
    return operation.model_copy(update={"group": None})


def strip_operation_groups(
    operations: Sequence[ToolProvenanceOperation],
) -> tuple[ToolProvenanceOperation, ...]:
    """Return operations with all edit-group metadata removed."""
    return tuple(_operation_without_group(operation) for operation in operations)


def stamp_operation_group(
    operations: Sequence[ToolProvenanceOperation],
    *,
    kind: str,
    focuses: Sequence[str | None] = (),
    group_id: str | None = None,
) -> tuple[ToolProvenanceOperation, ...]:
    """Return operations stamped as one complete contiguous edit group."""
    operation_tuple = tuple(_require_operation_instance(op) for op in operations)
    if not operation_tuple:
        return ()
    focus_tuple = tuple(focuses)
    if focus_tuple and len(focus_tuple) != len(operation_tuple):
        raise ValueError("Operation group focuses must match operation count")
    if not focus_tuple:
        focus_tuple = (None,) * len(operation_tuple)
    group_id = group_id or uuid.uuid4().hex
    size = len(operation_tuple)
    return tuple(
        operation.model_copy(
            update={
                "group": OperationGroupMarker(
                    kind=kind,
                    id=group_id,
                    index=index,
                    size=size,
                    focus=focus,
                )
            }
        )
        for index, (operation, focus) in enumerate(
            zip(operation_tuple, focus_tuple, strict=True)
        )
    )


def _operation_group_markers_match(
    operations: Sequence[ToolProvenanceOperation],
    start: int,
    stop: int,
    marker: OperationGroupMarker,
) -> bool:
    if start < 0 or stop > len(operations) or start >= stop:
        return False

    for offset, operation in enumerate(operations[start:stop]):
        operation_marker = operation.group
        if (
            operation_marker is None
            or operation_marker.kind != marker.kind
            or operation_marker.id != marker.id
            or operation_marker.size != marker.size
            or operation_marker.index != offset
        ):
            return False
    return True


def operation_group_range(
    operations: Sequence[ToolProvenanceOperation],
    operation_index: int,
    *,
    kind: str | None = None,
) -> tuple[int, int] | None:
    """Return the complete contiguous group range containing an operation."""
    if not 0 <= operation_index < len(operations):
        return None
    marker = operations[operation_index].group
    if marker is None or (kind is not None and marker.kind != kind):
        return None

    start = operation_index - marker.index
    stop = start + marker.size
    if not _operation_group_markers_match(operations, start, stop, marker):
        return None

    for neighbor_index in (start - 1, stop):
        if 0 <= neighbor_index < len(operations):
            neighbor_marker = operations[neighbor_index].group
            if (
                neighbor_marker is not None
                and neighbor_marker.kind == marker.kind
                and neighbor_marker.id == marker.id
            ):
                return None
    return start, stop


def restamp_operation_groups(
    operations: Sequence[ToolProvenanceOperation],
) -> tuple[ToolProvenanceOperation, ...]:
    """Return operations with complete groups copied to fresh group identities.

    Broken or partial group markers are stripped. This is useful when operations are
    pasted into another provenance chain, where preserving the copied group's id could
    make adjacent pasted copies look like one malformed group.
    """
    operation_tuple = tuple(_require_operation_instance(op) for op in operations)
    output = list(operation_tuple)
    index = 0
    while index < len(operation_tuple):
        marker = operation_tuple[index].group
        if marker is None:
            index += 1
            continue
        start = index - marker.index
        stop = start + marker.size
        if start == index and _operation_group_markers_match(
            operation_tuple,
            start,
            stop,
            marker,
        ):
            group_operations = operation_tuple[start:stop]
            output[start:stop] = stamp_operation_group(
                tuple(
                    _operation_without_group(operation)
                    for operation in group_operations
                ),
                kind=marker.kind,
                focuses=tuple(
                    None if operation.group is None else operation.group.focus
                    for operation in group_operations
                ),
            )
            index = stop
            continue
        output[index] = _operation_without_group(operation_tuple[index])
        index += 1
    return tuple(output)


def strip_partial_operation_groups(
    operations: Sequence[ToolProvenanceOperation],
) -> tuple[ToolProvenanceOperation, ...]:
    """Strip markers from broken or partial groups while preserving complete ones."""
    operation_tuple = tuple(_require_operation_instance(op) for op in operations)
    output = list(operation_tuple)
    for index, operation in enumerate(operation_tuple):
        if operation.group is None:
            continue
        group = operation_group_range(operation_tuple, index)
        if group is None:
            output[index] = _operation_without_group(operation)
    return tuple(output)


def operations_expression_code(
    operations: Sequence[ToolProvenanceOperation],
    input_name: str,
    *,
    source_name: str | None = None,
) -> str:
    """Return chained expression code for structured operations.

    ``source_name`` is the public/source array name passed to operations that need
    context beyond the transformed input, such as coordinate-order restoration.
    Operations that require control flow or multiple statements raise
    :class:`NotImplementedError`; use their :meth:`ToolProvenanceOperation.replay_code`
    instead.
    """
    if not operations:
        return ""
    source_name = input_name if source_name is None else source_name
    expression = input_name
    for operation in operations:
        expression = operation.expression_code(
            _expression_receiver_code(expression),
            source_name=source_name,
        )
    return expression


def operation_from_console_call(call: ConsoleCall) -> ToolProvenanceOperation | None:
    """Return the structured operation represented by a console call, if known."""
    _ensure_operation_catalog_loaded()
    for operation_type in _OPERATION_TYPES.values():
        operation = operation_type.from_console_call(call)
        if operation is not None:
            return operation
    return None


def _single_assign_coord(call: ConsoleCall) -> tuple[typing.Any, typing.Any] | None:
    if (
        call.has_extra_tracked_inputs
        or call.dataarray_method != "assign_coords"
        or call.accessor_path
        or len(call.args) > 1
    ):
        return None
    coords = _console_mapping_values(call.args, call.kwargs, mapping_kwargs=("coords",))
    if coords is None:
        return None
    if len(coords) != 1:
        return None
    return next(iter(coords.items()))


def _scalar_coord_value(value: typing.Any) -> bool:
    encoded = encode_provenance_value(value)
    decoded = decode_provenance_value(encoded)
    return not isinstance(decoded, (Mapping, Sequence)) or isinstance(decoded, str)


def _coerce_nullable_hashable_tuple_field(
    value: typing.Any,
) -> tuple[Hashable, ...] | None:
    if value is None:
        return None
    return ToolProvenanceOperation._coerce_hashable_tuple_field(value)


def _coerce_nullable_hashable_field(value: typing.Any) -> Hashable | None:
    if value is None:
        return None
    return ToolProvenanceOperation._coerce_hashable_field(value)


def _coerce_hashable_pair_field(value: typing.Any) -> tuple[Hashable, Hashable]:
    return typing.cast(
        "tuple[Hashable, Hashable]",
        ToolProvenanceOperation._coerce_hashable_tuple_field(value, expected_len=2),
    )


def _coerce_int_mapping_field(value: typing.Any) -> dict[Hashable, int]:
    return typing.cast(
        "dict[Hashable, int]",
        ToolProvenanceOperation._coerce_hashable_mapping_field(value, value_coerce=int),
    )


def _coerce_float_mapping_field(value: typing.Any) -> dict[Hashable, float]:
    return typing.cast(
        "dict[Hashable, float]",
        ToolProvenanceOperation._coerce_hashable_mapping_field(
            value, value_coerce=float
        ),
    )


def _coerce_hashable_mapping_field(value: typing.Any) -> dict[Hashable, Hashable]:
    return typing.cast(
        "dict[Hashable, Hashable]",
        ToolProvenanceOperation._coerce_hashable_mapping_field(
            value, value_coerce=_normalize_provenance_hashable
        ),
    )


def _coerce_float_sequence_mapping_field(
    value: typing.Any,
) -> dict[Hashable, list[float]]:
    return typing.cast(
        "dict[Hashable, list[float]]",
        ToolProvenanceOperation._coerce_hashable_mapping_field(
            value, value_coerce=_coerce_float_sequence
        ),
    )


ProvenanceHashable: typing.TypeAlias = typing.Annotated[
    Hashable,
    pydantic.BeforeValidator(ToolProvenanceOperation._coerce_hashable_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

NullableProvenanceHashable: typing.TypeAlias = typing.Annotated[
    Hashable | None,
    pydantic.BeforeValidator(_coerce_nullable_hashable_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

NullableProvenanceHashableTuple: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, ...] | None,
    pydantic.BeforeValidator(_coerce_nullable_hashable_tuple_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceHashableTuple: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, ...],
    pydantic.BeforeValidator(ToolProvenanceOperation._coerce_hashable_tuple_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceHashablePair: typing.TypeAlias = typing.Annotated[
    tuple[Hashable, Hashable],
    pydantic.BeforeValidator(_coerce_hashable_pair_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, typing.Any],
    pydantic.BeforeValidator(ToolProvenanceOperation._coerce_hashable_mapping_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceIntMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, int],
    pydantic.BeforeValidator(_coerce_int_mapping_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceFloatMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, float],
    pydantic.BeforeValidator(_coerce_float_mapping_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceHashableMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, Hashable],
    pydantic.BeforeValidator(_coerce_hashable_mapping_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]

ProvenanceFloatSequenceMapping: typing.TypeAlias = typing.Annotated[
    dict[Hashable, list[float]],
    pydantic.BeforeValidator(_coerce_float_sequence_mapping_field),
    pydantic.PlainSerializer(
        ToolProvenanceOperation._json_encode_field, when_used="json"
    ),
]


def _require_operation_instance(
    operation: ToolProvenanceOperation,
) -> ToolProvenanceOperation:
    """Require a fully constructed operation instance for runtime use."""
    if not isinstance(operation, ToolProvenanceOperation):
        raise TypeError(
            "Runtime provenance APIs accept ToolProvenanceOperation instances "
            "only. Use parse_tool_provenance_operation() when deserializing mappings."
        )
    return operation


def parse_tool_provenance_operation(
    value: ToolProvenanceOperation | Mapping[str, typing.Any],
) -> ToolProvenanceOperation:
    """Parse one serialized operation payload.

    This is the deserialize boundary for saved JSON. Runtime call sites should build
    concrete operation instances directly instead of passing raw mappings around.
    """
    if isinstance(value, ToolProvenanceOperation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Serialized provenance operations must be mappings")

    op = value.get("op")
    if not isinstance(op, str):
        raise TypeError("Serialized provenance operations must include a string `op`")

    _ensure_operation_catalog_loaded()
    operation_type = _OPERATION_TYPES.get(op)
    if operation_type is None:
        raise ValueError(
            f"Unknown provenance operation {op!r}; "
            f"expected one of {sorted(_OPERATION_TYPES)}"
        )
    payload = dict(value)
    if op == "script_code" and isinstance(payload.get("code"), str):
        payload["code"] = _migrate_legacy_nonuniform_restore_code(payload["code"])
    return operation_type.model_validate(payload)


def _is_legacy_coord_order_cleanup(
    operation: ToolProvenanceOperation,
) -> bool:
    """Identify cosmetic coordinate sorting retained only for saved-data parsing."""
    if _operation_is(operation, "sort_coord_order"):
        return True
    return _operation_is(operation, "script_code") and (
        operation.derivation_entry().label == _SORT_COORD_ORDER_DERIVATION_LABEL
    )


def _normalize_script_code_operations(
    spec: ToolProvenanceSpec,
) -> ToolProvenanceSpec:
    active_name = spec.active_name
    if spec.kind != "script" or active_name is None:
        return spec
    current_name: str | None = (
        active_name
        if any(script_input.name == active_name for script_input in spec.script_inputs)
        else None
    )
    if spec.seed_code is not None:
        current_name = _script_codes_output_name(
            (spec.seed_code,),
            active_name=active_name,
            current_name=current_name,
        )

    normalized_operations: list[ToolProvenanceOperation] = []
    for operation in spec.operations:
        if _operation_is(operation, "script_code"):
            operation_code = getattr(operation, "code", None)
            structured = _structured_operation_from_script_code(
                operation,
                current_name=current_name,
            )
            normalized_operations.append(structured)
            if (
                _operation_is(structured, "script_code")
                and getattr(operation, "copyable", False)
                and operation_code is not None
            ):
                current_name = _script_codes_output_name(
                    (operation_code,),
                    active_name=active_name,
                    current_name=current_name,
                )
            continue
        normalized_operations.append(operation)

    operations = tuple(normalized_operations)
    if operations == spec.operations:
        return spec
    if spec.kind in {"script", "file"}:
        return spec.model_copy(
            update={
                "steps": tuple(
                    step.model_copy(update={"operation": operation})
                    for step, operation in zip(spec.steps, operations, strict=True)
                )
            }
        )
    return spec.model_copy(update={"source_operations": operations})


class FileDataSelection(pydantic.BaseModel):
    """Serializable selection of one displayable array from a loaded file object.

    New provenance stores stable Dataset variable names and DataTree node/variable
    pairs instead of the positional parsed-array index used by older workspaces.
    """

    kind: typing.Literal[
        "dataarray",
        "dataset_variable",
        "datatree_variable",
        "sequence_index",
        "parsed_index",
    ]
    value: typing.Any = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _migrate_legacy_datatree_path(cls, value: typing.Any) -> typing.Any:
        if not isinstance(value, Mapping) or value.get("kind") != "datatree_path":
            return value
        path = value.get("value")
        if not isinstance(path, str) or not path.startswith("/"):
            return value
        node_path, separator, variable = path.rpartition("/")
        if not separator:
            return value
        payload = dict(value)
        payload["kind"] = "datatree_variable"
        payload["value"] = (node_path or "/", variable)
        return payload

    @pydantic.model_validator(mode="after")
    def _validate_value(self) -> typing.Self:
        if self.kind == "dataarray":
            if self.value is not None:
                raise ValueError("dataarray file selections must not define a value")
            return self
        if self.kind == "dataset_variable":
            value = _normalize_provenance_hashable(decode_provenance_value(self.value))
            if value != self.value:
                return self.model_copy(update={"value": value})
            return self
        if self.kind == "datatree_variable":
            if (
                isinstance(self.value, str | bytes)
                or not isinstance(self.value, Sequence)
                or len(self.value) != 2
            ):
                raise ValueError(
                    "datatree file selections must define a node path and variable"
                )
            node_path, variable = self.value
            if not isinstance(node_path, str) or not node_path.startswith("/"):
                raise ValueError("datatree node selections must use an absolute path")
            variable = _normalize_provenance_hashable(decode_provenance_value(variable))
            normalized = (node_path, variable)
            if normalized != self.value:
                return self.model_copy(update={"value": normalized})
            return self

        value = int(self.value) if isinstance(self.value, np.integer) else self.value
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("file selection index must be non-negative")
        if value != self.value:
            return self.model_copy(update={"value": value})
        return self

    @pydantic.field_serializer("value", when_used="json")
    def _serialize_value(self, value: typing.Any) -> typing.Any:
        if self.kind == "dataset_variable":
            return _encode_provenance_hashable(value)
        if self.kind == "datatree_variable":
            node_path, variable = value
            return [node_path, _encode_provenance_hashable(variable)]
        return value


class FileReplayCall(pydantic.BaseModel):
    """Serializable call information used to reload file-backed provenance."""

    kind: typing.Literal["erlab_loader", "callable"]
    target: str
    kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    selection: FileDataSelection
    cast_float64: bool = False

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_serialized_shape(cls, value: typing.Any) -> typing.Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        selected_index = payload.pop("selected_index", None)
        if "selection" not in payload and selected_index is not None:
            payload["selection"] = {
                "kind": "parsed_index",
                "value": selected_index,
            }
        return payload

    @pydantic.model_validator(mode="after")
    def _validate_replay_call(self) -> typing.Self:
        if not self.target:
            raise ValueError("target must not be empty")
        if any(not isinstance(key, str) for key in self.kwargs):
            raise TypeError("file replay kwargs must use string keys")
        return self


class ReplayStage(pydantic.BaseModel):
    """Legacy schema-v2 transformation stage.

    New provenance must use :class:`ReplayStep`.  This model remains public only so
    saved schema-v2 workspaces and callers constructing their old payloads continue to
    deserialize.  The compatibility conversion is centralized in
    :meth:`ToolProvenanceSpec._migrate_legacy_replay_shape`; remove this class together
    with that converter when schema-v2 workspace support is retired.
    """

    source_kind: _SourceKind
    operations: tuple[pydantic.SerializeAsAny[ToolProvenanceOperation], ...] = ()

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.field_validator("operations", mode="before")
    @classmethod
    def _validate_operations(
        cls, value: typing.Any
    ) -> tuple[ToolProvenanceOperation, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized replay stage operations must be a sequence")
        operations = tuple(
            parse_tool_provenance_operation(
                typing.cast("ToolProvenanceOperation | Mapping[str, typing.Any]", item)
            )
            for item in value
        )
        return tuple(
            operation
            for operation in operations
            if not _is_legacy_coord_order_cleanup(operation)
        )

    @pydantic.model_validator(mode="after")
    def _validate_live_operations(self) -> typing.Self:
        if any(not operation.live_applicable for operation in self.operations):
            raise TypeError(
                "legacy replay stages cannot contain script-only operations"
            )
        return self

    @classmethod
    def from_source_spec(cls, source: ToolProvenanceSpec) -> ReplayStage:
        live_source = require_live_source_spec(source)
        if live_source is None:
            raise TypeError("source must not be None")
        return cls(
            source_kind=typing.cast("_SourceKind", live_source.kind),
            operations=live_source.operations,
        )


class _SourceViewOperation(ToolProvenanceOperation):
    op: typing.Literal["source_view"] = "source_view"
    source_kind: _SourceKind

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        return _starting_data_for_source_kind(self.source_kind, data)

    def derivation_label(self) -> str:
        if self.source_kind == "selection":
            return "Start from selected parent ImageTool data"
        if self.source_kind == "public_data":
            return "Start from current parent ImageTool public data"
        return "Start from current parent ImageTool data"

    def expression_code(
        self, input_name: str, *, source_name: str | None = None
    ) -> str:
        input_expression = f"{input_name}.copy(deep=False)"
        if self.source_kind == "full_data":
            return input_expression
        raise NotImplementedError

    def statement_code(
        self,
        input_name: str,
        *,
        output_name: str,
        source_name: str | None = None,
    ) -> str:
        return _dynamic_nonuniform_restore_replay_code(
            input_name,
            output_name=output_name,
            copy_input=True,
        )


class _LegacyReplayContext(pydantic.BaseModel):
    """Schema-v2 parent context retained only for exact legacy operation replay.

    Schema-v2 stage operations shared one ``parent_data`` value, while top-level
    operations received their own input as ``parent_data``. New operations are
    self-contained and new replay steps do not populate this model. Keeping the
    compatibility state on migrated steps makes the removal boundary explicit.
    """

    index: int
    input_policy: _ReplayInputPolicy

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class ReplayStep(pydantic.BaseModel):
    """One operation in the canonical ordered durable replay pipeline."""

    operation: pydantic.SerializeAsAny[ToolProvenanceOperation]
    input_policy: _ReplayInputPolicy | None = None
    context_names: tuple[str, ...] = pydantic.Field(
        default=(),
        exclude_if=lambda value: not value,
    )
    legacy_context: _LegacyReplayContext | None = pydantic.Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.field_validator("operation", mode="before")
    @classmethod
    def _validate_operation(cls, value: typing.Any) -> ToolProvenanceOperation:
        return parse_tool_provenance_operation(value)

    @pydantic.field_validator("context_names", mode="before")
    @classmethod
    def _validate_context_names(cls, value: typing.Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Replay step context names must be a sequence")
        names: list[str] = []
        for item in value:
            name = _validate_active_name(item)
            if name is None:
                raise ValueError("Replay step context names must not be None")
            if name not in names:
                names.append(name)
        return tuple(names)

    @pydantic.model_validator(mode="after")
    def _validate_step(self) -> typing.Self:
        if self.input_policy is not None and self.legacy_context is not None:
            raise ValueError(
                "Replay steps cannot define both current and legacy input policies"
            )
        if self.input_policy is not None and not self.operation.live_applicable:
            raise TypeError("Source-backed replay steps require live operations")
        return self

    @classmethod
    def from_source_spec(cls, source: ToolProvenanceSpec) -> tuple[ReplayStep, ...]:
        """Lower one live source into independently movable durable steps."""
        live_source = require_live_source_spec(source)
        if live_source is None:
            raise TypeError("source must not be None")
        input_policy: _ReplayInputPolicy = (
            "current" if live_source.kind == "full_data" else "restored"
        )
        if not live_source.operations:
            if live_source.kind in {"full_data", "public_data"}:
                return ()
            return (
                cls(
                    operation=_SourceViewOperation(
                        source_kind=typing.cast("_SourceKind", live_source.kind)
                    ),
                    input_policy="current",
                ),
            )
        return tuple(
            cls(operation=operation, input_policy=input_policy)
            for operation in live_source.operations
        )


def _steps_for_legacy_operations_update(
    current_steps: Sequence[ReplayStep],
    operations: Sequence[ToolProvenanceOperation],
) -> tuple[ReplayStep, ...]:
    """Translate the schema-v2 operations alias without dropping step metadata.

    Old extensions may insert, remove, replace, or reorder operation objects through
    ``model_copy(update={"operations": ...})``. Match retained operations by identity
    and then value, and transfer the remaining step metadata positionally to genuine
    replacements. New insertions receive a new unannotated step.

    Remove this function with the ``operations`` model-copy alias.
    """
    old_steps = tuple(current_steps)
    new_operations = tuple(operations)
    old_operation_payloads = tuple(
        step.operation.model_dump(mode="json") for step in old_steps
    )
    new_operation_payloads = tuple(
        operation.model_dump(mode="json") for operation in new_operations
    )
    unmatched_old = list(range(len(old_steps)))
    matched_old_by_new: dict[int, int] = {}

    for new_index, operation in enumerate(new_operations):
        for old_index in unmatched_old:
            if old_steps[old_index].operation is operation:
                matched_old_by_new[new_index] = old_index
                unmatched_old.remove(old_index)
                break

    for new_index, operation_payload in enumerate(new_operation_payloads):
        if new_index in matched_old_by_new:
            continue
        candidates = [
            old_index
            for old_index in unmatched_old
            if old_operation_payloads[old_index] == operation_payload
        ]
        if not candidates:
            continue
        first = old_steps[candidates[0]]
        if any(
            (
                candidate.input_policy,
                candidate.context_names,
                candidate.legacy_context,
            )
            != (first.input_policy, first.context_names, first.legacy_context)
            for candidate in (old_steps[index] for index in candidates[1:])
        ):
            raise ValueError(
                "Cannot match duplicate operations with different replay metadata; "
                "update `steps` explicitly"
            )
        old_index = candidates[0]
        matched_old_by_new[new_index] = old_index
        unmatched_old.remove(old_index)

    unmatched_new = [
        index for index in range(len(new_operations)) if index not in matched_old_by_new
    ]
    matched_old_by_new.update(dict(zip(unmatched_new, unmatched_old, strict=False)))

    return tuple(
        ReplayStep(operation=operation)
        if (old_index := matched_old_by_new.get(new_index)) is None
        else old_steps[old_index].model_copy(update={"operation": operation})
        for new_index, operation in enumerate(new_operations)
    )


class FileLoadSource(pydantic.BaseModel):
    """Serializable file origin used by saved file-backed provenance."""

    path: str
    loader_label: str
    loader_text: str
    kwargs_text: str
    replay_call: FileReplayCall | None = None
    load_code: str | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: typing.Any) -> str:
        return str(value)

    @pydantic.field_validator("load_code", mode="before")
    @classmethod
    def _migrate_load_code(cls, value: typing.Any) -> typing.Any:
        if value is None or isinstance(value, str):
            return _migrate_legacy_nonuniform_restore_code(value)
        return value


class ScriptInput(pydantic.BaseModel):
    """Named input captured by script or multi-tool provenance.

    ``name`` is the immutable replay variable, ``label`` is the historical display
    label, ``node_uid`` and ``node_snapshot_token`` identify the live manager input
    that was used, ``data_role`` selects its durable source or displayed view, and
    ``provenance_spec`` stores the historical replay source used when that live input
    is unavailable. When omitted, ``label`` defaults to ``name``.
    """

    name: str
    label: str = ""
    node_uid: str | None = None
    node_snapshot_token: str | None = None
    data_role: ScriptInputDataRole = pydantic.Field(
        default="displayed",
        exclude_if=lambda value: value == "displayed",
    )
    provenance_spec: dict[str, typing.Any] | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _default_label_from_name(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, Mapping) and (
            "label" not in value or value.get("label") is None
        ):
            value = dict(value)
            value["label"] = value.get("name")
        return value

    @pydantic.field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: typing.Any) -> str:
        name = _validate_active_name(value)
        if name is None:
            raise ValueError("script input name must not be None")
        return name

    @pydantic.field_validator("label", mode="before")
    @classmethod
    def _validate_label(cls, value: typing.Any) -> str:
        if not isinstance(value, str):
            raise TypeError("script input label must be a string")
        label = " ".join(value.split())
        if not label:
            raise ValueError("script input label must not be empty")
        return label

    @pydantic.field_validator("node_uid", mode="before")
    @classmethod
    def _validate_node_uid(cls, value: typing.Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @pydantic.field_validator("node_snapshot_token", mode="before")
    @classmethod
    def _validate_node_snapshot_token(cls, value: typing.Any) -> str | None:
        if value is None:
            return None
        token = str(value)
        if not token:
            raise ValueError("script input snapshot token must not be empty")
        return token

    @pydantic.field_validator("provenance_spec", mode="before")
    @classmethod
    def _validate_provenance_spec(
        cls, value: typing.Any
    ) -> dict[str, typing.Any] | None:
        if value is None:
            return None
        if isinstance(value, ToolProvenanceSpec):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(
            "script input provenance must be a ToolProvenanceSpec or mapping"
        )

    def parsed_provenance_spec(self) -> ToolProvenanceSpec | None:
        return parse_tool_provenance_spec(self.provenance_spec)


def _script_input_reference_text(script_input: ScriptInput) -> str:
    if script_input.label == script_input.name:
        return script_input.name
    return f"{script_input.name} from {script_input.label}"


class _ScriptContextBinding(pydantic.BaseModel):
    """Hidden binding from the current script output to names used by later code."""

    operation_index: int
    names: tuple[str, ...]

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.field_validator("operation_index", mode="before")
    @classmethod
    def _validate_operation_index(cls, value: typing.Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("script context operation index must be an integer")
        if value < 0:
            raise ValueError("script context operation index must be non-negative")
        return value

    @pydantic.field_validator("names", mode="before")
    @classmethod
    def _validate_names(cls, value: typing.Any) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("script context names must be a sequence")
        names: list[str] = []
        for item in value:
            name = _validate_active_name(item)
            if name is None:
                raise ValueError("script context names must not be None")
            if name not in names:
                names.append(name)
        if not names:
            raise ValueError("script context names must not be empty")
        return tuple(names)


class ToolProvenanceSpec(pydantic.BaseModel):
    """Saved provenance recipe for ImageTool data.

    Live child-tool refresh uses single-parent specs from :func:`full_data`,
    :func:`public_data`, or :func:`selection`. Durable reload and copied code use replay
    specs. The ``kind`` field selects the replay representation, not the user-visible
    origin: ``file_load_source`` is the file-origin capability, and ``steps`` is the
    single ordered operation sequence used by both ``file`` and ``script`` replay
    specs. ``script`` specs may also include multi-input ``script_inputs`` for console
    or UI actions that combine several ImageTools.
    Deserialize saved payloads with
    :func:`parse_tool_provenance_spec`.

    A spec records exact operation arguments for live refresh, runtime replay, copied
    code, and derivation display. Manager children opened from ImageTool cursor or bin
    selections refresh by replaying those explicit arguments; legacy selection
    bindings are converted to source specs once for compatibility.
    """

    schema_version: typing.Literal[3] = 3
    kind: typing.Literal["full_data", "public_data", "selection", "script", "file"]
    start_label: str | None = None
    seed_code: str | None = None
    active_name: str | None = None
    source_operations: tuple[pydantic.SerializeAsAny[ToolProvenanceOperation], ...] = (
        pydantic.Field(
            default=(),
            validation_alias=pydantic.AliasChoices("operations", "source_operations"),
            serialization_alias="operations",
        )
    )
    steps: tuple[ReplayStep, ...] = ()
    file_load_source: FileLoadSource | None = None
    script_inputs: tuple[ScriptInput, ...] = ()

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        serialize_by_alias=True,
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_serialized_shape(cls, value: typing.Any) -> typing.Any:
        if not isinstance(value, Mapping):
            return value
        payload = cls._migrate_legacy_replay_shape(value)
        if payload.get("kind") == "script" and "active_name" not in payload:
            raise ValueError("script provenance specs must define `active_name`")
        return payload

    @staticmethod
    def _migrate_legacy_replay_shape(
        value: Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Convert schema-v1/v2 containers to the flat schema-v3 step sequence.

        This is the sole compatibility boundary for saved replay stages, indexed
        script context bindings, and operations whose schema-v2 implementation read
        ``parent_data``. New runtime authoring never creates ``legacy_context`` values.
        """
        payload = dict(value)
        legacy_keys = (
            "replay_stages" in payload
            or "script_context_bindings" in payload
            or (
                payload.get("kind") in {"script", "file"}
                and "steps" not in payload
                and "operations" in payload
            )
        )
        schema_version = payload.get("schema_version", 2 if legacy_keys else 3)
        if schema_version == 3 and not legacy_keys:
            return payload
        if schema_version not in {1, 2, 3}:
            return payload

        kind = payload.get("kind")
        raw_operations = payload.get("operations") or ()
        if kind == "file" and raw_operations:
            raise ValueError("file provenance specs cannot define operations")
        raw_bindings = payload.pop("script_context_bindings", ()) or ()
        bindings_by_index: dict[int, tuple[str, ...]] = {}
        for raw_binding in raw_bindings:
            binding = (
                raw_binding
                if isinstance(raw_binding, _ScriptContextBinding)
                else _ScriptContextBinding.model_validate(raw_binding)
            )
            existing_names = bindings_by_index.get(binding.operation_index, ())
            bindings_by_index[binding.operation_index] = tuple(
                dict.fromkeys((*existing_names, *binding.names))
            )

        # Schema-v2 ScriptCodeOperation payloads can load names such as
        # ``parent_data`` from these indexed bindings. Attach the names to the
        # operation itself so saved scripts retain that behavior without keeping the
        # old parallel collection in the active model.

        steps: list[dict[str, typing.Any] | ReplayStep] = []
        raw_stages = payload.pop("replay_stages", ()) or ()
        if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, Sequence):
            raise TypeError("Serialized replay stages must be a sequence")
        for stage_index, raw_stage in enumerate(raw_stages):
            stage = (
                raw_stage
                if isinstance(raw_stage, ReplayStage)
                else ReplayStage.model_validate(raw_stage)
            )
            input_policy: _ReplayInputPolicy = (
                "current" if stage.source_kind == "full_data" else "restored"
            )
            if not stage.operations:
                if stage.source_kind == "full_data":
                    continue
                steps.append(
                    {
                        "operation": _SourceViewOperation(
                            source_kind=stage.source_kind
                        ),
                    }
                )
                continue
            steps.extend(
                {
                    "operation": operation,
                    "legacy_context": {
                        "index": stage_index,
                        "input_policy": input_policy,
                    },
                }
                for operation in stage.operations
            )

        if kind in {"script", "file"}:
            if isinstance(raw_operations, (str, bytes)) or not isinstance(
                raw_operations, Sequence
            ):
                raise TypeError("Serialized provenance operations must be a sequence")
            for index, raw_operation in enumerate(raw_operations):
                operation = parse_tool_provenance_operation(raw_operation)
                step: dict[str, typing.Any] = {
                    "operation": operation,
                    "context_names": bindings_by_index.get(index, ()),
                }
                if not _operation_is(operation, "script_code"):
                    # Schema-v2 applied each top-level structured operation with the
                    # operation input as both ``data`` and ``parent_data``. Retain that
                    # exact call contract for third-party operation subclasses until
                    # schema-v2 workspace compatibility is removed.
                    step["legacy_context"] = {
                        "index": len(raw_stages) + index,
                        "input_policy": "current",
                    }
                steps.append(step)
            payload["operations"] = ()
        payload["steps"] = steps
        payload["schema_version"] = 3
        return payload

    @pydantic.field_validator("active_name", mode="before")
    @classmethod
    def _validate_active_name_field(cls, value: typing.Any) -> str | None:
        return _validate_active_name(value)

    @pydantic.field_validator("seed_code", mode="before")
    @classmethod
    def _migrate_seed_code(cls, value: typing.Any) -> typing.Any:
        if value is None or isinstance(value, str):
            return _migrate_legacy_nonuniform_restore_code(value)
        return value

    @pydantic.field_validator("source_operations", mode="before")
    @classmethod
    def _validate_operations(
        cls, value: typing.Any
    ) -> tuple[ToolProvenanceOperation, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized provenance operations must be a sequence")
        operations = tuple(
            parse_tool_provenance_operation(
                typing.cast("ToolProvenanceOperation | Mapping[str, typing.Any]", item)
            )
            for item in value
        )
        return tuple(
            operation
            for operation in operations
            if not _is_legacy_coord_order_cleanup(operation)
        )

    @pydantic.field_validator("steps", mode="before")
    @classmethod
    def _validate_steps(cls, value: typing.Any) -> tuple[ReplayStep, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized replay steps must be a sequence")
        steps = tuple(
            item if isinstance(item, ReplayStep) else ReplayStep.model_validate(item)
            for item in value
        )
        return tuple(
            step for step in steps if not _is_legacy_coord_order_cleanup(step.operation)
        )

    @pydantic.field_validator("script_inputs", mode="before")
    @classmethod
    def _validate_script_inputs(cls, value: typing.Any) -> tuple[ScriptInput, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized script inputs must be a sequence")
        return tuple(
            item if isinstance(item, ScriptInput) else ScriptInput.model_validate(item)
            for item in value
        )

    @pydantic.model_validator(mode="after")
    def _validate_kind_fields(self) -> typing.Self:
        if self.kind == "script":
            if self.start_label is None:
                raise ValueError("script provenance specs must define `start_label`")
            if self.source_operations:
                raise ValueError("script provenance specs must define replay steps")
            return self
        if self.kind == "file":
            if self.start_label is None:
                raise ValueError("file provenance specs must define `start_label`")
            if self.seed_code is None:
                raise ValueError("file provenance specs must define `seed_code`")
            if self.active_name is None:
                raise ValueError("file provenance specs must define `active_name`")
            if self.file_load_source is None:
                raise ValueError("file provenance specs must define `file_load_source`")
            if self.file_load_source.replay_call is None:
                raise ValueError("file provenance specs must define `replay_call`")
            if self.source_operations:
                raise ValueError("file provenance specs must define replay steps")
            if any(step.context_names for step in self.steps):
                raise ValueError("file provenance steps cannot define script context")
            return self
        if (
            self.start_label is not None
            or self.seed_code is not None
            or self.active_name is not None
            or self.file_load_source is not None
            or self.steps
            or self.script_inputs
        ):
            raise ValueError(
                "Only script or file provenance specs may define `start_label`, "
                "`seed_code`, `active_name`, `file_load_source`, `steps`, or "
                "`script_inputs`"
            )
        return self

    @property
    def operations(self) -> tuple[ToolProvenanceOperation, ...]:
        """Return the ordered operations for this live source or durable recipe."""
        if self.kind in {"script", "file"}:
            return tuple(step.operation for step in self.steps)
        return self.source_operations

    @property
    def script_context_bindings(self) -> tuple[_ScriptContextBinding, ...]:
        """Compatibility projection of explicit replay-step context names."""
        if self.kind != "script":
            return ()
        return tuple(
            _ScriptContextBinding(operation_index=index, names=step.context_names)
            for index, step in enumerate(self.steps)
            if step.context_names
        )

    @property
    def is_live_source(self) -> bool:
        return self.kind in {"full_data", "public_data", "selection"} and all(
            operation.live_applicable for operation in self.operations
        )

    def model_copy(
        self,
        *,
        update: Mapping[str, typing.Any] | None = None,
        deep: bool = False,
    ) -> typing.Self:
        """Copy while translating the schema-v2 ``operations`` update alias.

        ``BaseModel.model_copy`` does not validate aliases.  Older manager extensions
        may still update the serialized ``operations`` name directly, so keep this
        small compatibility translation until schema-v2 extension support is removed.
        New runtime code updates ``source_operations`` or ``steps`` explicitly.
        """
        normalized = dict(update or {})
        if "replay_stages" in normalized:
            raw_stages = normalized.pop("replay_stages")
            top_steps = tuple(
                step
                for step in self.steps
                if step.input_policy is None and step.legacy_context is None
            )
            legacy_payload = self._migrate_legacy_replay_shape(
                {
                    "schema_version": 2,
                    "kind": self.kind,
                    "operations": tuple(step.operation for step in top_steps),
                    "replay_stages": raw_stages,
                    "script_context_bindings": tuple(
                        _ScriptContextBinding(
                            operation_index=index,
                            names=step.context_names,
                        )
                        for index, step in enumerate(top_steps)
                        if step.context_names
                    ),
                }
            )
            normalized["steps"] = tuple(
                ReplayStep.model_validate(step)
                for step in legacy_payload.get("steps", ())
            )
        if "operations" in normalized:
            operations = tuple(
                _require_operation_instance(operation)
                for operation in normalized.pop("operations")
            )
            if self.kind in {"script", "file"}:
                current_steps = self._validate_steps(
                    normalized.get("steps", self.steps)
                )
                normalized["steps"] = _steps_for_legacy_operations_update(
                    current_steps,
                    operations,
                )
            else:
                normalized["source_operations"] = operations
        if "steps" in normalized:
            normalized["steps"] = self._validate_steps(normalized["steps"])
        if "source_operations" in normalized:
            normalized["source_operations"] = self._validate_operations(
                normalized["source_operations"]
            )
        return super().model_copy(update=normalized, deep=deep)

    def append_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Append operation instances to the spec.

        Runtime code should pass operation instances from this module. Saved mappings
        should be normalized with :func:`parse_tool_provenance_spec` before calling
        this method.
        """
        appended = tuple(_require_operation_instance(op) for op in operations)
        if self.kind in {"script", "file"}:
            return self.model_copy(
                update={
                    "steps": (
                        *self.steps,
                        *(ReplayStep(operation=operation) for operation in appended),
                    )
                }
            )
        return self.model_copy(
            update={"source_operations": (*self.source_operations, *appended)}
        )

    def drop_trailing_rename(self) -> ToolProvenanceSpec:
        if not self.operations or not _operation_is(self.operations[-1], "rename"):
            return self
        if self.kind in {"script", "file"}:
            return self.model_copy(update={"steps": self.steps[:-1]})
        return self.model_copy(
            update={"source_operations": self.source_operations[:-1]}
        )

    def append_replacement_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Replace a final rename, if present, then append new operation instances."""
        return self.drop_trailing_rename().append_operations(*operations)

    def append_display_operation(
        self, operation: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Append a live display operation without making a final name stale."""
        operation = _require_operation_instance(operation)
        if not self.is_live_source:
            raise TypeError("Display operations can only be appended to live sources")
        operations = self.operations
        if operations and _operation_is(operations[-1], "rename"):
            return self.model_copy(
                update={
                    "source_operations": (
                        *operations[:-1],
                        operation,
                        operations[-1],
                    )
                }
            )
        return self.append_operations(operation)

    def append_final_rename(self, name: str) -> ToolProvenanceSpec:
        return self.drop_trailing_rename().append_operations(
            _operation_instance("rename", name=name)
        )

    def append_replay_stage(self, source: ToolProvenanceSpec) -> ToolProvenanceSpec:
        """Compatibility alias for appending a live source as flat replay steps."""
        if self.kind not in {"file", "script"}:
            raise TypeError("Replay steps can only be appended to replay provenance")
        return self.model_copy(
            update={"steps": (*self.steps, *ReplayStep.from_source_spec(source))}
        )

    def _operation_for_ref(
        self, ref: _ProvenanceStepRef
    ) -> ToolProvenanceOperation | None:
        if ref.kind != "operation" or ref.operation_index is None:
            return None
        if 0 <= ref.operation_index < len(self.operations):
            return self.operations[ref.operation_index]
        return None

    def _replace_operation_ref(
        self,
        ref: _ProvenanceStepRef,
        replacements: Sequence[ToolProvenanceOperation],
    ) -> ToolProvenanceSpec:
        if ref.kind != "operation" or ref.operation_index is None:
            raise ValueError("Expected an operation provenance row reference")
        return self._replace_operation_range_ref(
            ref,
            ref.operation_index,
            ref.operation_index + 1,
            replacements,
        )

    def _replace_operation_range_ref(
        self,
        ref: _ProvenanceStepRef,
        start: int,
        stop: int,
        replacements: Sequence[ToolProvenanceOperation],
    ) -> ToolProvenanceSpec:
        if ref.kind != "operation" or ref.operation_index is None:
            raise ValueError("Expected an operation provenance row reference")
        if start < 0 or stop <= start:
            raise ValueError("Expected a non-empty operation range")
        replacement_ops = tuple(_require_operation_instance(op) for op in replacements)
        if self.kind not in {"script", "file"}:
            operations = list(self.source_operations)
            operations[start:stop] = replacement_ops
            return self.model_copy(update={"source_operations": tuple(operations)})

        replaced_steps = self.steps[start:stop]
        first_step = replaced_steps[0]
        common_input_policy = (
            first_step.input_policy
            if all(
                step.input_policy == first_step.input_policy for step in replaced_steps
            )
            else None
        )
        common_legacy_context = (
            first_step.legacy_context
            if all(
                step.legacy_context == first_step.legacy_context
                for step in replaced_steps
            )
            else None
        )
        replacement_steps = tuple(
            ReplayStep(
                operation=operation,
                input_policy=common_input_policy,
                context_names=first_step.context_names if index == 0 else (),
                legacy_context=common_legacy_context,
            )
            for index, operation in enumerate(replacement_ops)
        )
        steps = list(self.steps)
        steps[start:stop] = replacement_steps
        return self.model_copy(update={"steps": tuple(steps)})

    @staticmethod
    def _operation_is_reorderable(operation: ToolProvenanceOperation) -> bool:
        if not _operation_is(operation, "script_code"):
            return operation.live_applicable
        return bool(
            getattr(operation, "copyable", False) and getattr(operation, "code", None)
        )

    @staticmethod
    def _operation_reorder_entry(
        operation: ToolProvenanceOperation,
    ) -> DerivationEntry | None:
        """Return the stable reorder-dialog entry for one stored operation.

        This intentionally excludes only operations that are structurally internal or
        explicitly hidden. Context-dependent display cleanup, such as eliding a no-op
        transpose or squeeze, must not change reorder boundaries.
        """
        entry = operation.derivation_entry()
        if _operation_is(operation, "script_code"):
            if not getattr(operation, "visible", True):
                return None
            if entry.code in {
                "derived = derived.isel()",
                "derived = derived.qsel()",
                "derived = derived.sel()",
            }:
                return None
        elif _operation_is(operation, "qsel", "isel", "sel"):
            if not getattr(operation, "decoded_kwargs", {}):
                return None
        elif _operation_is(operation, "rename", "source_view"):
            return None
        return entry

    def _reorderable_operation_blocks(
        self,
        *,
        fixed_refs: Collection[_ProvenanceStepRef] = (),
    ) -> tuple[_ProvenanceReorderBlock, ...]:
        """Return the structural atomic operation blocks in the flat recipe.

        Operation identity, grouping, and reorder visibility come directly from the
        stored provenance. Display streamlining and the hierarchical ``display_rows()``
        representation are deliberately not part of reorder planning.
        """
        fixed_ref_set = set(fixed_refs)
        operations = self.operations
        reorder_entries = tuple(
            self._operation_reorder_entry(operation) for operation in operations
        )
        blocks: list[_ProvenanceReorderBlock] = []
        operation_index = 0
        while operation_index < len(operations):
            operation = operations[operation_index]
            group_range = operation_group_range(operations, operation_index)
            if operation.group is not None and group_range is None:
                operation_index += 1
                continue
            start, stop = (
                (operation_index, operation_index + 1)
                if group_range is None
                else group_range
            )
            operation_index = stop
            entries = tuple(
                entry
                for index in range(start, stop)
                if (entry := reorder_entries[index]) is not None
            )
            if not entries:
                continue
            block_refs = {
                _ProvenanceStepRef("operation", operation_index=index)
                for index in range(start, stop)
            }
            if block_refs & fixed_ref_set or any(
                not self._operation_is_reorderable(item)
                for item in operations[start:stop]
            ):
                continue
            blocks.append(
                _ProvenanceReorderBlock(
                    _ProvenanceReorderBlockRef(start, stop),
                    entries,
                )
            )
        return tuple(blocks)

    def _reorder_sections(
        self,
        *,
        fixed_refs: Collection[_ProvenanceStepRef] = (),
    ) -> tuple[_ProvenanceReorderSection, ...]:
        """Return contiguous flat operation sections safe to permute."""
        segments: list[list[_ProvenanceReorderBlock]] = []
        for block in self._reorderable_operation_blocks(fixed_refs=fixed_refs):
            if not segments or segments[-1][-1].ref.stop != block.ref.start:
                segments.append([block])
            else:
                segments[-1].append(block)
        movable_segments = [segment for segment in segments if len(segment) >= 2]
        return tuple(
            _ProvenanceReorderSection(
                ref=_ProvenanceReorderSectionRef(
                    segment[0].ref.start,
                    segment[-1].ref.stop,
                ),
                label=(
                    "Steps"
                    if len(movable_segments) == 1
                    else f"Steps — section {segment_index}"
                ),
                blocks=tuple(segment),
            )
            for segment_index, segment in enumerate(movable_segments, start=1)
        )

    def _reorder_operation_blocks(
        self,
        sections: Sequence[_ProvenanceReorderSection],
        orders: Mapping[
            _ProvenanceReorderSectionRef,
            Sequence[_ProvenanceReorderBlockRef],
        ],
    ) -> ToolProvenanceSpec:
        """Return a validated spec with allowed provenance blocks permuted."""
        expected_section_refs = tuple(section.ref for section in sections)
        if len(set(expected_section_refs)) != len(expected_section_refs):
            raise ValueError("Provenance reorder sections must be unique")
        if set(orders) != set(expected_section_refs):
            raise ValueError("Provenance reorder plan does not match its sections")

        allowed_block_refs = {
            block.ref for block in self._reorderable_operation_blocks()
        }
        index_order = list(range(len(self.operations)))
        occupied_ranges: list[tuple[int, int]] = []
        for section in sections:
            section_ref = section.ref
            if not (0 <= section_ref.start < section_ref.stop <= len(self.operations)):
                raise ValueError("Provenance reorder section is out of range")
            if any(
                section_ref.start < stop and start < section_ref.stop
                for start, stop in occupied_ranges
            ):
                raise ValueError("Provenance reorder sections must not overlap")
            occupied_ranges.append((section_ref.start, section_ref.stop))

            expected_blocks = tuple(block.ref for block in section.blocks)
            if not set(expected_blocks) <= allowed_block_refs:
                raise ValueError(
                    "Provenance reorder blocks must represent movable displayed steps"
                )
            cursor = section_ref.start
            for block_ref in expected_blocks:
                if (
                    block_ref.start != cursor
                    or not block_ref.start < block_ref.stop <= section_ref.stop
                ):
                    raise ValueError(
                        "Provenance reorder blocks must partition their section"
                    )
                operation = self.operations[block_ref.start]
                group_range = operation_group_range(
                    self.operations,
                    block_ref.start,
                )
                if operation.group is None:
                    if block_ref.stop != block_ref.start + 1:
                        raise ValueError(
                            "Ungrouped provenance blocks must contain one operation"
                        )
                elif group_range != (block_ref.start, block_ref.stop):
                    raise ValueError(
                        "Grouped provenance blocks must contain the complete group"
                    )
                cursor = block_ref.stop
            if cursor != section_ref.stop or len(set(expected_blocks)) != len(
                expected_blocks
            ):
                raise ValueError(
                    "Provenance reorder blocks must partition their section"
                )

            requested_blocks = tuple(orders[section_ref])
            if len(requested_blocks) != len(expected_blocks) or set(
                requested_blocks
            ) != set(expected_blocks):
                raise ValueError(
                    "Provenance reorder plan must contain every block exactly once"
                )
            index_order[section_ref.start : section_ref.stop] = [
                index
                for block_ref in requested_blocks
                for index in range(block_ref.start, block_ref.stop)
            ]

        if self.kind in {"script", "file"}:
            candidate = self.model_copy(
                update={"steps": tuple(self.steps[index] for index in index_order)}
            )
        else:
            candidate = self.model_copy(
                update={
                    "source_operations": tuple(
                        self.source_operations[index] for index in index_order
                    )
                }
            )
        return type(self).model_validate(candidate.model_dump(mode="python"))

    def _prefix_through_ref(self, ref: _ProvenanceStepRef) -> ToolProvenanceSpec:
        """Return a spec replaying through the referenced displayed row."""
        if ref.kind in {"start", "file_load"}:
            if self.kind in {"file", "script"}:
                updates: dict[str, typing.Any] = {"steps": ()}
                if output_name := self._script_seed_output_name():
                    updates["active_name"] = output_name
                return self.model_copy(update=updates)
            if self.kind in {"full_data", "public_data", "selection"}:
                return self.model_copy(update={"source_operations": ()})

        if ref.kind != "operation" or ref.operation_index is None:
            raise ValueError("This provenance row cannot be replayed as a prefix")
        end = ref.operation_index + 1
        if self.kind in {"script", "file"}:
            return self.model_copy(update={"steps": self.steps[:end]})
        return self.model_copy(
            update={"source_operations": self.source_operations[:end]}
        )

    def _prefix_before_ref(self, ref: _ProvenanceStepRef) -> ToolProvenanceSpec:
        """Return a spec replaying to the input of the referenced operation row."""
        if ref.kind != "operation" or ref.operation_index is None:
            return self._prefix_through_ref(ref)
        if self.kind in {"script", "file"}:
            return self.model_copy(update={"steps": self.steps[: ref.operation_index]})
        return self.model_copy(
            update={"source_operations": self.source_operations[: ref.operation_index]}
        )

    def _script_seed_output_name(self) -> str | None:
        """Return the variable produced by this script spec's seed code."""
        if self.kind != "script" or self.seed_code is None or self.active_name is None:
            return None
        return _script_codes_output_name(
            (self.seed_code,),
            active_name=self.active_name,
            current_name=None,
        )

    def _start_entry(self) -> DerivationEntry:
        if self.kind in {"full_data", "public_data"}:
            return DerivationEntry(
                "Start from current parent ImageTool data",
                None,
                False,
            )
        if self.kind == "selection":
            return DerivationEntry(
                "Start from selected parent ImageTool data",
                None,
                False,
            )
        if self.kind == "file":
            return DerivationEntry(
                typing.cast("str", self.start_label),
                None,
                False,
            )
        return DerivationEntry(
            typing.cast("str", self.start_label),
            None,
            False,
        )

    @staticmethod
    def _starting_data_for_kind(
        source_kind: _SourceKind,
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        return _starting_data_for_source_kind(source_kind, parent_data)

    @staticmethod
    def _streamlined_operations(
        source_kind: typing.Literal["full_data", "public_data", "selection"],
        operations: Sequence[ToolProvenanceOperation],
        *,
        parent_data: xr.DataArray | None = None,
    ) -> tuple[ToolProvenanceOperation, ...]:
        return tuple(
            operation
            for _, operation in ToolProvenanceSpec._streamlined_operation_refs(
                source_kind,
                operations,
                parent_data=parent_data,
            )
        )

    @staticmethod
    def _streamlined_operation_refs(
        source_kind: typing.Literal["full_data", "public_data", "selection"],
        operations: Sequence[ToolProvenanceOperation],
        *,
        parent_data: xr.DataArray | None = None,
        include_hidden_script_code: bool = False,
    ) -> tuple[tuple[int, ToolProvenanceOperation], ...]:
        context = _ProvenanceDisplayContext.from_source(source_kind, parent_data)

        streamlined: list[tuple[int, ToolProvenanceOperation]] = []
        for index, operation in enumerate(operations):
            hide_operation = False

            if _operation_is(operation, "script_code"):
                entry = operation.derivation_entry()
                hide_operation = (
                    not getattr(operation, "visible", True)
                    and not include_hidden_script_code
                ) or entry.code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                }
            # Rule 1: drop empty selection operations.
            elif _operation_is(operation, "qsel", "isel", "sel"):
                hide_operation = not getattr(operation, "decoded_kwargs", {})
            # Rule 2: hide source-view compatibility operations.
            elif _operation_is(operation, "source_view"):
                hide_operation = True
            # Rule 2c: hide final whole-array name changes unless arbitrary script code
            # follows and may observe the DataArray name.
            elif _operation_is(operation, "rename"):
                hide_operation = (
                    _is_whole_array_rename_entry(operation.derivation_entry())
                    and bool(streamlined)
                    and not any(
                        _operation_is(later_operation, "script_code")
                        for later_operation in operations[index + 1 :]
                    )
                )
            # Rule 3: drop transpose calls that do not change dimension order.
            elif _operation_is(operation, "transpose") and context.dims is not None:
                operation_dims = getattr(operation, "dims", None)
                target_dims = (
                    tuple(operation_dims)
                    if operation_dims is not None
                    else tuple(reversed(context.dims))
                )
                hide_operation = target_dims == context.dims
            # Rule 4: drop squeeze calls that would not remove singleton dimensions.
            elif _operation_is(operation, "squeeze") and context.sizes is not None:
                operation_dims = getattr(operation, "dims", None)
                if operation_dims is None:
                    hide_operation = not any(
                        size == 1 for size in context.sizes.values()
                    )
                else:
                    hide_operation = not any(
                        context.sizes.get(dim) == 1 for dim in operation_dims
                    )
            # Rule 5: drop nonuniform restoration when it would not change dimensions.
            elif (
                _operation_is(operation, "restore_nonuniform_dims")
                and context.dims is not None
            ):
                hide_operation = not context.dims_may_restore_nonuniform(context.dims)
            if not hide_operation:
                streamlined.append((index, operation))

            # Rule 6: keep anything ambiguous. Metadata-only display never executes
            # operations; if a step has unknown effects, later no-op checks become
            # conservative and preserve rows/code.
            context = context.advance(operation)

        return tuple(streamlined)

    def _display_operations(
        self, *, parent_data: xr.DataArray | None = None
    ) -> tuple[ToolProvenanceOperation, ...]:
        if self.kind in {"script", "file"}:
            raise TypeError("Script and file provenance use custom display filtering")
        return self._streamlined_operations(
            typing.cast(
                "typing.Literal['full_data', 'public_data', 'selection']",
                self.kind,
            ),
            self.operations,
            parent_data=parent_data,
        )

    def to_replay_spec(self) -> ToolProvenanceSpec:
        """Return the durable replay form for this spec.

        Replay specs are the canonical, composable form used for derivation metadata,
        copied code, workspace save/load, and manager dependency status. Structured
        file provenance remains structured so runtime reloads can replay typed
        operations without ``exec``. Live-source operations are kept structured inside
        script replay specs so editable manager rows remain typed after composition.
        Live ImageTool refresh uses the original single-parent spec via
        :func:`require_live_source_spec`.
        """
        if self.kind in {"script", "file"}:
            return self

        return ToolProvenanceSpec(
            kind="script",
            start_label=self._start_entry().label,
            seed_code=_default_seed_code_for_operations(self.operations),
            active_name="derived",
            file_load_source=self.file_load_source,
            steps=ReplayStep.from_source_spec(self),
        )

    def apply(self, parent_data: xr.DataArray) -> xr.DataArray:
        require_live_source_spec(self)
        data = self._starting_data_for_kind(
            typing.cast(
                "typing.Literal['full_data', 'public_data', 'selection']",
                self.kind,
            ),
            parent_data,
        )
        for operation in self.operations:
            data = operation._apply_schema_v2(data, parent_data=parent_data)
        if self.kind == "selection":
            # Coordinate dictionary order is presentation state, not a replay step.
            # Normalize it once at the live selection boundary so it cannot split or
            # constrain the operation sequence shown in the reorder dialog.
            data = erlab.utils.array.sort_coord_order(data, parent_data.coords.keys())
        return data

    def derivation_entries(self) -> list[DerivationEntry]:
        entries: list[DerivationEntry] = [self._start_entry()]
        if self.kind == "script":
            entries.extend(
                DerivationEntry(
                    f"Use {_script_input_reference_text(script_input)}",
                    None,
                    False,
                )
                for script_input in self.script_inputs
            )
        for operation in self.operations:
            entry = operation.derivation_entry()
            entries.append(entry)
        return entries

    def _script_graph_code(self, *, display: bool) -> str | None:
        if not self.operations:
            return None
        from erlab.interactive.imagetool._provenance._graph import (
            ReplayGraphError,
            compile_replay_graph,
            emit_replay_code,
        )

        try:
            graph = compile_replay_graph(self, display=display)
            return emit_replay_code(
                graph,
                output_name=typing.cast("str", self.active_name),
            )
        except ReplayGraphError:
            return None

    def _code_lines_from_entries(
        self, entries: Sequence[DerivationEntry]
    ) -> list[str] | None:
        codes = []
        for entry in entries:
            if entry.code is None:
                return None
            codes.append(entry.code)
        return codes

    def _code_fallback_entries(
        self, *, parent_data: xr.DataArray | None = None
    ) -> list[DerivationEntry]:
        if self.kind != "script":
            return self.display_entries(parent_data=parent_data)

        entries = [self._start_entry()]
        entries.extend(
            DerivationEntry(
                f"Use {_script_input_reference_text(script_input)}",
                None,
                False,
            )
            for script_input in self.script_inputs
        )

        entries.extend(
            operation.derivation_entry()
            for _, operation in self._streamlined_operation_refs(
                "full_data",
                self.operations,
                parent_data=parent_data,
                include_hidden_script_code=True,
            )
        )
        return entries

    def derivation_code(self) -> str | None:
        prefix: str | None = None
        if self.kind == "script" and (
            graph_code := self._script_graph_code(display=True)
        ):
            return graph_code
        if self.kind in {"script", "file"}:
            prefix = self.seed_code
        entries = self._code_fallback_entries()
        step_codes = self._code_lines_from_entries(entries[1:])
        if step_codes is None:
            return None
        if prefix is None and self.kind != "script":
            if not step_codes:
                return None
            prefix = _default_seed_code_for_operations(self.operations)
        if prefix is None and not step_codes:
            return None
        return _simplify_display_code(
            "\n".join(part for part in (prefix, *step_codes) if part)
        )

    def display_rows(
        self,
        *,
        parent_data: xr.DataArray | None = None,
        scope: typing.Literal["display", "source"] = "display",
    ) -> list[_ProvenanceDisplayRow]:
        """Return streamlined derivation rows for UI, editing, and copy-code output.

        The display path hides internal ImageTool normalization steps while keeping the
        recorded replay steps available through :meth:`derivation_entries`.
        """

        def with_scope(
            row: _ProvenanceDisplayRow,
        ) -> _ProvenanceDisplayRow:
            return replace(
                row,
                scope=scope,
                children=tuple(with_scope(child) for child in row.children),
            )

        def scoped_rows(
            rows: list[_ProvenanceDisplayRow],
        ) -> list[_ProvenanceDisplayRow]:
            if scope == "display":
                return rows
            return [with_scope(row) for row in rows]

        def input_history_rows(
            rows: Sequence[_ProvenanceDisplayRow],
            path: tuple[int, ...],
        ) -> tuple[_ProvenanceDisplayRow, ...]:
            return tuple(
                replace(
                    row,
                    script_input_path=path + row.script_input_path,
                    children=input_history_rows(row.children, path),
                )
                for row in rows
            )

        start_ref = _ProvenanceStepRef(
            "file_load"
            if self.kind == "file"
            or (self.kind == "script" and self.file_load_source is not None)
            else "start"
        )
        rows = [
            _ProvenanceDisplayRow(
                self._start_entry(),
                edit_ref=start_ref if start_ref.kind == "file_load" else None,
                replay_ref=start_ref,
            )
        ]

        if self.kind == "script":
            for index, script_input in enumerate(self.script_inputs):
                rows.append(
                    _ProvenanceDisplayRow(
                        DerivationEntry(
                            f"Use {_script_input_reference_text(script_input)}",
                            None,
                            False,
                        ),
                        replay_ref=_ProvenanceStepRef(
                            "script_input", script_input_index=index
                        ),
                        children=(
                            ()
                            if (input_spec := script_input.parsed_provenance_spec())
                            is None
                            else input_history_rows(
                                input_spec.display_rows(),
                                (index,),
                            )
                        ),
                    )
                )
            rows.extend(
                _ProvenanceDisplayRow(
                    operation.derivation_entry(),
                    edit_ref=(
                        None
                        if (
                            _operation_is(operation, "script_code")
                            and (
                                getattr(operation, "code", None) is None
                                or not getattr(operation, "copyable", False)
                            )
                        )
                        else _ProvenanceStepRef(
                            "operation", operation_index=operation_index
                        )
                    ),
                    replay_ref=_ProvenanceStepRef(
                        "operation", operation_index=operation_index
                    ),
                )
                for operation_index, operation in self._streamlined_operation_refs(
                    "full_data",
                    self.operations,
                )
            )
            return scoped_rows(rows)
        if self.kind == "file":
            rows.extend(
                _ProvenanceDisplayRow(
                    operation.derivation_entry(),
                    edit_ref=_ProvenanceStepRef(
                        "operation", operation_index=operation_index
                    ),
                    replay_ref=_ProvenanceStepRef(
                        "operation", operation_index=operation_index
                    ),
                )
                for operation_index, operation in self._streamlined_operation_refs(
                    "full_data",
                    self.operations,
                    parent_data=parent_data,
                )
            )
            return scoped_rows(rows)

        rows.extend(
            _ProvenanceDisplayRow(
                operation.derivation_entry(),
                edit_ref=_ProvenanceStepRef(
                    "operation", operation_index=operation_index
                ),
                replay_ref=_ProvenanceStepRef(
                    "operation", operation_index=operation_index
                ),
            )
            for operation_index, operation in self._streamlined_operation_refs(
                self.kind,
                self.operations,
                parent_data=parent_data,
            )
        )
        return scoped_rows(rows)

    def display_entries(
        self, *, parent_data: xr.DataArray | None = None
    ) -> list[DerivationEntry]:
        """Return streamlined derivation entries for UI and copy-code output."""
        return [row.entry for row in self.display_rows(parent_data=parent_data)]

    def display_code(self, *, parent_data: xr.DataArray | None = None) -> str | None:
        """Return streamlined replay code for UI and clipboard actions.

        The display path preserves exact live-source behavior while omitting user-facing
        no-op and normalization steps from copied provenance code.
        """
        prefix: str | None = None
        if self.kind == "script" and (
            graph_code := self._script_graph_code(display=True)
        ):
            return graph_code
        if self.kind in {"script", "file"}:
            prefix = self.seed_code

        entries = self._code_fallback_entries(parent_data=parent_data)
        step_codes = self._code_lines_from_entries(entries[1:])
        if step_codes is None:
            return None

        if prefix is None and self.kind != "script":
            if not step_codes:
                return None
            prefix = _default_seed_code_for_operations(self.operations)
        if prefix is None and not step_codes:
            return None
        if not step_codes and prefix == _DEFAULT_REPLAY_SEED_CODE:
            return None
        inline_targets = (
            {"derived"}
            if self.kind == "script" and self.active_name != "derived"
            else None
        )
        return _simplify_display_code(
            "\n".join(part for part in (prefix, *step_codes) if part),
            inline_targets=inline_targets,
        )


def parse_tool_provenance_spec(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> ToolProvenanceSpec | None:
    """Parse a serialized provenance payload into a validated spec instance.

    This is the deserialize boundary for saved tool and workspace metadata. Runtime
    authoring code should pass :class:`ToolProvenanceSpec` instances directly.
    """
    if value is None:
        return None
    if isinstance(value, ToolProvenanceSpec):
        return value
    return _normalize_script_code_operations(ToolProvenanceSpec.model_validate(value))


def has_file_load_source(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> bool:
    """Return whether a spec records a file origin, independent of replay kind."""
    spec = parse_tool_provenance_spec(value)
    return spec is not None and spec.file_load_source is not None


def iter_operation_refs(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> Iterator[tuple[_ProvenanceStepRef, ToolProvenanceOperation]]:
    """Yield operation references in replay order across all operation stores."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return
    for operation_index, operation in enumerate(spec.operations):
        yield (
            _ProvenanceStepRef("operation", operation_index=operation_index),
            operation,
        )


def script_input_dependency_refs(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> tuple[ScriptInputDependencyRef, ...]:
    """Return all live manager dependency references stored in script inputs."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return ()

    refs: list[ScriptInputDependencyRef] = []

    def _collect(current: ToolProvenanceSpec) -> None:
        for script_input in current.script_inputs:
            if script_input.node_uid:
                refs.append(
                    ScriptInputDependencyRef(
                        name=script_input.name,
                        label=script_input.label,
                        node_uid=script_input.node_uid,
                        node_snapshot_token=script_input.node_snapshot_token,
                        data_role=script_input.data_role,
                    )
                )
            nested = script_input.parsed_provenance_spec()
            if nested is not None:
                _collect(nested)

    _collect(spec)
    return tuple(refs)


def rebase_script_input_node_uids(
    value: ToolProvenanceSpec | Mapping[str, typing.Any],
    uid_map: Mapping[str, str],
) -> ToolProvenanceSpec:
    """Return ``value`` with script input node UIDs remapped recursively."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        raise TypeError("Expected provenance spec")
    if not uid_map or not spec.script_inputs:
        return spec

    def _rebase(current: ToolProvenanceSpec) -> ToolProvenanceSpec:
        if not current.script_inputs:
            return current

        changed = False
        script_inputs: list[ScriptInput] = []
        for script_input in current.script_inputs:
            updates: dict[str, typing.Any] = {}
            if script_input.node_uid is not None and script_input.node_uid in uid_map:
                updates["node_uid"] = uid_map[script_input.node_uid]

            nested = script_input.parsed_provenance_spec()
            if nested is not None:
                rebased_nested = _rebase(nested)
                if rebased_nested != nested:
                    updates["provenance_spec"] = rebased_nested.model_dump(mode="json")

            if updates:
                changed = True
                script_inputs.append(script_input.model_copy(update=updates))
            else:
                script_inputs.append(script_input)

        if not changed:
            return current
        return current.model_copy(update={"script_inputs": tuple(script_inputs)})

    return _rebase(spec)


def to_replay_provenance_spec(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> ToolProvenanceSpec | None:
    """Parse ``value`` and normalize it into canonical replay provenance."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return None
    return spec.to_replay_spec()


def mark_promoted_1d_source(data: xr.DataArray) -> xr.DataArray:
    """Return ``data`` tagged as originating from a promoted 1D source."""
    data.attrs = dict(data.attrs)
    data.attrs[_PROMOTED_1D_SOURCE_ATTR] = True
    return data


def compose_display_provenance(
    parent: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    source_spec: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    parent_data: xr.DataArray | None = None,
) -> ToolProvenanceSpec | None:
    """Compose streamlined display provenance from a live source spec."""
    parent_spec = to_replay_provenance_spec(parent)
    source = require_live_source_spec(source_spec)
    if source is None:
        return parent_spec
    if (
        parent_spec is not None
        and source.kind in {"full_data", "public_data"}
        and not source.operations
    ):
        return parent_spec
    if (
        direct_replay_input_name(parent_spec) is not None
        and parent_data is not None
        and parent_data.attrs.get(_PROMOTED_1D_SOURCE_ATTR, False)
        and source.kind == "selection"
    ):
        saw_squeeze = False
        for operation in source.operations:
            if _operation_is(operation, "qsel", "isel", "sel"):
                if getattr(operation, "decoded_kwargs", {}):
                    break
                continue
            if (
                _operation_is(operation, "squeeze")
                and getattr(operation, "dims", None) is None
                and not getattr(operation, "drop", False)
            ):
                saw_squeeze = True
                continue
            break
        else:
            if saw_squeeze:
                return parent_spec

    source_kind = typing.cast(
        "typing.Literal['full_data', 'public_data', 'selection']",
        source.kind,
    )
    display_operations = tuple(
        operation
        for _, operation in ToolProvenanceSpec._streamlined_operation_refs(
            source_kind,
            source.operations,
            parent_data=parent_data,
        )
    )
    local_spec = ToolProvenanceSpec(
        kind="script",
        start_label=source._start_entry().label,
        seed_code=_DEFAULT_REPLAY_SEED_CODE,
        active_name="derived",
        # Display provenance is already rooted at the displayed parent expression.
        # Source restoration belongs to executable full provenance, not this
        # streamlined projection.
        steps=tuple(
            ReplayStep(operation=operation) for operation in display_operations
        ),
    )
    if parent_spec is None:
        return local_spec
    return compose_full_provenance(
        parent_spec,
        local_spec.model_copy(update={"seed_code": None}),
    )


def direct_replay_input_name(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> str | None:
    """Return a direct input expression for simple replay seeds.

    This only applies to non-default single-line seeds such as watched variables.
    Generic replay aliases like ``derived = data`` continue to use ``derived`` so
    existing non-watched code generation remains stable.
    """
    spec = to_replay_provenance_spec(value)
    if (
        spec is None
        or spec.operations
        or spec.script_inputs
        or spec.seed_code is None
        or spec.seed_code == _DEFAULT_REPLAY_SEED_CODE
    ):
        return None

    prefix = "derived = "
    if not spec.seed_code.startswith(prefix):
        return None

    name = spec.seed_code.removeprefix(prefix).strip()
    if not name.isidentifier() or keyword.iskeyword(name):
        try:
            parsed = ast.parse(name, mode="eval")
        except SyntaxError:
            return None
        match parsed.body:
            case ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=base_name),
                    attr="astype",
                ),
                args=[
                    ast.Attribute(
                        value=ast.Name(id="np"),
                        attr="float64",
                    )
                ],
                keywords=[],
            ) if base_name.isidentifier() and not keyword.iskeyword(base_name):
                return name
            case _:
                return None
    return name


def replay_input_name(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> str | None:
    spec = to_replay_provenance_spec(value)
    if spec is None:
        return None
    direct_name = direct_replay_input_name(spec)
    if direct_name is not None:
        return direct_name
    return spec.active_name or "derived"


def _as_script_replay_spec(spec: ToolProvenanceSpec) -> ToolProvenanceSpec:
    """Return a generated-code replay spec for composition fallbacks."""
    if spec.kind == "script":
        return spec
    if spec.kind == "file":
        return ToolProvenanceSpec(
            kind="script",
            start_label=typing.cast("str", spec.start_label),
            seed_code=spec.seed_code,
            active_name=spec.active_name,
            file_load_source=spec.file_load_source,
            steps=spec.steps,
        )
    return spec.to_replay_spec()


def compose_full_provenance(
    parent: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    local: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    script_context_names: Sequence[str] = (),
) -> ToolProvenanceSpec | None:
    """Compose canonical full provenance from parent and local provenance.

    ``parent`` represents the replay provenance for the current input data. ``local``
    represents the additional steps performed by the current node. File-backed parents
    remain structured when composed with live-source specs so runtime reloads can avoid
    executing generated Python.
    """
    parent_value = parse_tool_provenance_spec(parent)
    local_value = parse_tool_provenance_spec(local)

    if parent_value is None:
        return None if local_value is None else local_value.to_replay_spec()
    if local_value is None:
        return parent_value.to_replay_spec()

    parent_replay = parent_value.to_replay_spec()
    with contextlib.suppress(TypeError):
        local_live = require_live_source_spec(local_value)
        if local_live is not None:
            if not local_live.operations:
                return parent_replay
            prefix_steps: tuple[ReplayStep, ...] = ()
            if parent_replay.kind == "script" and parent_replay.active_name not in {
                None,
                "derived",
            }:
                prefix_steps = (
                    ReplayStep(
                        operation=_operation_instance(
                            "script_code",
                            label="Use current parent output as the active data",
                            code=f"derived = {parent_replay.active_name}",
                            visible=False,
                        )
                    ),
                )
            return parent_replay.model_copy(
                update={
                    "active_name": "derived",
                    "steps": (
                        *parent_replay.steps,
                        *prefix_steps,
                        *ReplayStep.from_source_spec(local_live),
                    ),
                }
            )

    parent_spec = _as_script_replay_spec(parent_replay)
    local_spec = _as_script_replay_spec(local_value.to_replay_spec())

    normalized_context_names: list[str] = []
    for item in script_context_names:
        name = _validate_active_name(item)
        if name is None:
            raise ValueError("script context names must not be None")
        if name not in normalized_context_names:
            normalized_context_names.append(name)

    parent_legacy_indices = [
        step.legacy_context.index
        for step in parent_spec.steps
        if step.legacy_context is not None
    ]
    legacy_index_offset = max(parent_legacy_indices, default=-1) + 1
    local_steps = [
        (
            step
            if step.legacy_context is None
            else step.model_copy(
                update={
                    "legacy_context": step.legacy_context.model_copy(
                        update={
                            "index": step.legacy_context.index + legacy_index_offset
                        }
                    )
                }
            )
        )
        for step in local_spec.steps
    ]
    if local_spec.seed_code:
        seed_code: str | None = local_spec.seed_code
        if seed_code == _DEFAULT_REPLAY_SEED_CODE:
            parent_input = replay_input_name(parent_spec)
            if parent_input == "derived" or parent_spec.active_name == "derived":
                seed_code = None
            elif parent_input is not None:
                seed_code = f"derived = {parent_input}"
        if seed_code is not None:
            local_steps.insert(
                0,
                ReplayStep(
                    operation=_operation_instance(
                        "script_code",
                        label=typing.cast("str", local_spec.start_label),
                        code=seed_code,
                        visible=False,
                    )
                ),
            )
    elif local_spec.active_name == "derived":
        parent_input = replay_input_name(parent_spec)
        if (
            parent_input is not None
            and parent_input != "derived"
            and parent_spec.active_name not in {None, "derived"}
            and local_steps
        ):
            local_steps.insert(
                0,
                ReplayStep(
                    operation=_operation_instance(
                        "script_code",
                        label="Use current parent output as the active data",
                        code=f"derived = {parent_input}",
                        visible=False,
                    )
                ),
            )

    if normalized_context_names and local_value.kind == "script" and local_steps:
        first_step = local_steps[0]
        local_steps[0] = first_step.model_copy(
            update={
                "context_names": tuple(
                    dict.fromkeys(
                        (*normalized_context_names, *first_step.context_names)
                    )
                )
            }
        )

    return ToolProvenanceSpec(
        kind="script",
        start_label=typing.cast("str", parent_spec.start_label),
        seed_code=parent_spec.seed_code,
        active_name=local_spec.active_name or parent_spec.active_name,
        file_load_source=parent_spec.file_load_source or local_spec.file_load_source,
        script_inputs=(*parent_spec.script_inputs, *local_spec.script_inputs),
        steps=(*parent_spec.steps, *local_steps),
    )


def require_live_source_spec(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> ToolProvenanceSpec | None:
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return None
    if not spec.is_live_source:
        raise TypeError(
            "source_spec must be a live ToolProvenanceSpec. Use a non-script spec "
            "whose operations support `apply()`."
        )
    return spec


def full_data(
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Build a spec that starts from the parent's full current data."""
    return ToolProvenanceSpec(kind="full_data").append_operations(*operations)


def public_data(
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Build a spec that starts from the parent's restored public data."""
    return ToolProvenanceSpec(kind="public_data").append_operations(*operations)


def selection(
    *operations: ToolProvenanceOperation,
) -> ToolProvenanceSpec:
    """Build a spec that starts from the parent's public selection model."""
    return ToolProvenanceSpec(kind="selection").append_operations(*operations)


def script(
    *operations: ToolProvenanceOperation,
    start_label: str,
    seed_code: str | None = None,
    active_name: str | None = None,
    file_load_source: FileLoadSource | None = None,
    steps: Sequence[ReplayStep] = (),
    replay_stages: Sequence[ReplayStage] = (),
    script_inputs: Sequence[ScriptInput] = (),
) -> ToolProvenanceSpec:
    """Build script provenance from code, structured steps, and named inputs."""
    if steps and replay_stages:
        raise ValueError("Use replay steps or legacy replay stages, not both")
    payload: dict[str, typing.Any] = {
        "kind": "script",
        "start_label": start_label,
        "seed_code": seed_code,
        "active_name": active_name,
        "file_load_source": file_load_source,
        "steps": tuple(steps),
        "script_inputs": tuple(script_inputs),
    }
    if replay_stages:
        # Constructor-level schema-v2 compatibility. Runtime authoring should pass
        # ReplayStep objects; remove this branch with ReplayStage support.
        payload["schema_version"] = 2
        payload["replay_stages"] = tuple(replay_stages)
        payload.pop("steps")
    return ToolProvenanceSpec.model_validate(payload).append_operations(*operations)


def file_load(
    *,
    start_label: str,
    seed_code: str,
    file_load_source: FileLoadSource,
    active_name: str = "derived",
    steps: Sequence[ReplayStep] = (),
    replay_stages: Sequence[ReplayStage] = (),
) -> ToolProvenanceSpec:
    """Build structured file-backed provenance for runtime reload."""
    if steps and replay_stages:
        raise ValueError("Use replay steps or legacy replay stages, not both")
    payload: dict[str, typing.Any] = {
        "kind": "file",
        "start_label": start_label,
        "seed_code": seed_code,
        "active_name": active_name,
        "file_load_source": file_load_source,
        "steps": tuple(steps),
    }
    if replay_stages:
        # Constructor-level schema-v2 compatibility; see script() above.
        payload["schema_version"] = 2
        payload["replay_stages"] = tuple(replay_stages)
        payload.pop("steps")
    return ToolProvenanceSpec.model_validate(payload)
