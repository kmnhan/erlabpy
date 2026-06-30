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
     and replayed through structured :class:`ReplayStage` operations.

   - Use :func:`script` for console-derived and other multi-input results. Script
     specs may reference any number of :class:`ScriptInput` records. Each input stores
     an immutable replay name, a historical display label, optional live manager node
     identity and ``node_snapshot_token``, and an optional nested provenance snapshot.

Each transformation is represented by an immutable
:class:`ToolProvenanceOperation` subclass whose serialized fields are safe to persist
in JSON. Persisted specs are the source of truth for workspace save/load. Runtime
reload, copied code, and shared-input deduplication compile those specs on demand
through :mod:`erlab.interactive.imagetool._replay_graph`; the graph itself is not
saved.

Operations are intentionally primitive: a user action that needs several public API
calls should normally remain several structured operations rather than falling back to
one raw script block. When those primitive operations should behave as one edit unit,
authoring code can stamp them with optional :class:`OperationGroupMarker` metadata.
Group markers are edit/copy metadata only. They do not change replay semantics, and
broken or partial groups can be stripped without changing the numerical operation list.

Manager children opened from an ImageTool cursor or bin selection keep the explicit
``qsel`` or ``isel`` arguments generated when the child is opened. Legacy
:class:`~erlab.interactive.imagetool.provenance.ImageToolSelectionSourceBinding`
payloads are materialized once into a normal source spec before refresh.

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
   using the recorded parameters. ``parent_data`` is provided when the operation needs
   access to parent coordinates or ordering.

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

10. Export the operation class from this module so runtime call sites can instantiate
   it directly.

11. Add tests that cover round-trip validation, :meth:`apply`, derivation text/code,
    console matching when supported, and any save/load or reload path that persists the
    new operation. For grouped operations, also cover full-group copy/paste, partial
    group stripping, group replacement/deletion, and generated-code execution.

Parsing of serialized payloads happens only through :func:`parse_tool_provenance_spec`
and :func:`parse_tool_provenance_operation`. Runtime authoring code should create specs
with :func:`full_data`, :func:`public_data`, :func:`selection`, :func:`file_load`, or
:func:`script`, then instantiate operation models from this module directly.
"""

from __future__ import annotations

__all__ = [
    "DerivationEntry",
    "FileDataSelection",
    "FileLoadSource",
    "FileLoadSourceStatus",
    "FileReplayCall",
    "OperationGroupMarker",
    "ReplayStage",
    "ScriptInput",
    "ScriptInputDependencyRef",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "can_reload_without_trust",
    "compose_display_provenance",
    "compose_full_provenance",
    "decode_provenance_value",
    "direct_replay_input_name",
    "encode_provenance_value",
    "file_load",
    "file_load_source_status",
    "full_data",
    "has_file_load_source",
    "iter_operation_refs",
    "mark_promoted_1d_source",
    "operation_group_range",
    "operations_expression_code",
    "parse_tool_provenance_spec",
    "public_data",
    "rebase_default_replay_input",
    "rebase_script_input_node_uids",
    "replay_file_provenance",
    "replay_script_provenance",
    "require_live_source_spec",
    "restamp_operation_groups",
    "script",
    "script_input_dependency_refs",
    "script_provenance_replayable",
    "selection",
    "stamp_operation_group",
    "strip_operation_groups",
    "strip_partial_operation_groups",
    "to_replay_provenance_spec",
    "uses_default_replay_input",
]

import ast
import base64
import contextlib
import importlib
import inspect
import keyword
import math
import pathlib
import typing
import uuid
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool import _replay_graph

_SourceKind: typing.TypeAlias = typing.Literal["full_data", "public_data", "selection"]
FileLoadSourceStatus: typing.TypeAlias = typing.Literal[
    "loadable",
    "no-file-load-source",
    "missing-file",
    "no-replay-call",
    "missing-loader",
]

_SLICE_MARKER = "__erlab_slice__"
_DATASET_MARKER = "__erlab_xarray_dataset__"
_FIT_DATASET_MARKER = "__erlab_xarray_lmfit_dataset__"
_DATAARRAY_MARKER = "__erlab_xarray_dataarray__"
_TUPLE_MARKER = "__erlab_tuple__"
_MAPPING_MARKER = "__erlab_mapping__"
_DEFAULT_REPLAY_SEED_CODE = "derived = data"
_PROMOTED_1D_SOURCE_ATTR = "_erlab_promoted_from_1d_source"
_SORT_COORD_ORDER_DERIVATION_LABEL = "Sort coordinates to parent order"
_SORT_COORD_ORDER_DERIVATION_CODE = (
    "derived = erlab.utils.array.sort_coord_order(derived, data.coords.keys())"
)
_SCRIPT_REPLAY_ALLOWED_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "complex": complex,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "ImportError": ImportError,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "slice": slice,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_SCRIPT_REPLAY_ALLOWED_IMPORT_ROOTS = {
    "erlab",
    "matplotlib",
    "numpy",
    "seaborn",
    "xarray",
}
_SCRIPT_REPLAY_FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_SCRIPT_REPLAY_FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
}


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
    stage_index: int | None = None
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
        if operation_name in {"sort_coord_order", "rename"}:
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


def _is_internal_sort_coord_order_entry(entry: DerivationEntry) -> bool:
    return (
        entry.label == _SORT_COORD_ORDER_DERIVATION_LABEL
        and entry.code == _SORT_COORD_ORDER_DERIVATION_CODE
    )


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


def _provenance_value_code(value: typing.Any) -> str:
    value = erlab.utils.misc._convert_to_native(value)
    if isinstance(value, np.ndarray):
        return f"np.array({_provenance_value_code(value.tolist())})"
    if isinstance(value, float):
        if math.isnan(value):
            return "np.nan"
        if math.isinf(value):
            return "np.inf" if value > 0 else "-np.inf"
        return repr(value)
    if isinstance(value, complex):
        return (
            f"complex({_provenance_value_code(float(value.real))}, "
            f"{_provenance_value_code(float(value.imag))})"
        )
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_provenance_value_code(item) for item in value) + "]"
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return (
            "("
            + ", ".join(_provenance_value_code(item) for item in value)
            + suffix
            + ")"
        )
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(
                f"{_provenance_value_code(key)}: {_provenance_value_code(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    raise TypeError(f"Cannot generate replay code for {type(value).__name__!r}")


def _provenance_numeric_array_code(values: typing.Any) -> str:
    values = np.asarray(values)
    if (
        values.ndim == 1
        and np.issubdtype(values.dtype, np.number)
        and not np.issubdtype(values.dtype, np.complexfloating)
        and np.all(np.isfinite(values.astype(np.float64, copy=False)))
    ):
        return erlab.interactive.utils.format_1d_numeric_array_code(values)
    return _provenance_value_code(values)


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


def _format_selection_expr(
    input_name: str, method: str, kwargs: Mapping[Hashable, typing.Any]
) -> str:
    if not kwargs:
        return f"{input_name}.{method}()"
    args = erlab.interactive.utils.format_kwargs(dict(kwargs))
    return f"{input_name}.{method}({args})"


def _format_selection_step(method: str, kwargs: Mapping[Hashable, typing.Any]) -> str:
    return f"derived = {_format_selection_expr('derived', method, kwargs)}"


def _starting_data_for_source_kind(
    source_kind: _SourceKind,
    parent_data: xr.DataArray,
) -> xr.DataArray:
    if source_kind == "full_data":
        return parent_data.copy(deep=False)
    return erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
        parent_data.copy(deep=False)
    )


def _validate_active_name(value: typing.Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("active_name must be a string or None")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError("active_name must be a valid Python identifier")
    return value


class _CurrentScopeNameCounter(ast.NodeVisitor):
    def __init__(
        self,
        target: str,
        contexts: tuple[type[ast.expr_context], ...],
        *,
        count_definition_names: bool = True,
    ) -> None:
        self.target = target
        self.contexts = contexts
        self.count_definition_names = count_definition_names
        self.count = 0

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.target and isinstance(node.ctx, self.contexts):
            self.count += 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.count_definition_names and ast.Store in self.contexts:
            self.count += node.name == self.target
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_expressions(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(typing.cast("ast.FunctionDef", node))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_expressions(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.count_definition_names and ast.Store in self.contexts:
            self.count += node.name == self.target
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword_arg in node.keywords:
            self.visit(keyword_arg)

    def _visit_argument_expressions(self, args: ast.arguments) -> None:
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *(arg for arg in (args.vararg, args.kwarg) if arg is not None),
        ):
            if arg.annotation is not None:
                self.visit(arg.annotation)


class _ScopedNameReplacer(ast.NodeTransformer):
    def __init__(self, target: str, replacement: ast.expr) -> None:
        self._target = target
        self._replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self._target:
            return _clone_expr(self._replacement)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        self._visit_argument_expressions(node.args)
        if node.returns is not None:
            node.returns = typing.cast("ast.expr", self.visit(node.returns))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(typing.cast("ast.FunctionDef", node))

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        self._visit_argument_expressions(node.args)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        node.bases = [typing.cast("ast.expr", self.visit(base)) for base in node.bases]
        node.keywords = [
            typing.cast("ast.keyword", self.visit(keyword_arg))
            for keyword_arg in node.keywords
        ]
        return node

    def _visit_argument_expressions(self, args: ast.arguments) -> None:
        args.defaults = [
            typing.cast("ast.expr", self.visit(default)) for default in args.defaults
        ]
        args.kw_defaults = [
            None if default is None else typing.cast("ast.expr", self.visit(default))
            for default in args.kw_defaults
        ]
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *(arg for arg in (args.vararg, args.kwarg) if arg is not None),
        ):
            if arg.annotation is not None:
                arg.annotation = typing.cast("ast.expr", self.visit(arg.annotation))


def _statement_load_count(stmt: ast.stmt, target: str) -> int:
    counter = _CurrentScopeNameCounter(target, (ast.Load,))
    counter.visit(stmt)
    return counter.count


def _statement_store_count(
    stmt: ast.stmt, target: str, *, count_definition_names: bool = False
) -> int:
    counter = _CurrentScopeNameCounter(
        target,
        (ast.Store, ast.Del),
        count_definition_names=count_definition_names,
    )
    counter.visit(stmt)
    return counter.count


class _NameReplacer(_ScopedNameReplacer):
    def __init__(self, target: str, replacement: ast.expr) -> None:
        super().__init__(target, replacement)


def _clone_expr(node: ast.expr) -> ast.expr:
    return ast.parse(ast.unparse(node), mode="eval").body


def _clone_stmt(node: ast.stmt) -> ast.stmt:
    return ast.parse(ast.unparse(node), mode="exec").body[0]


class _IdentifierReplacer(ast.NodeTransformer):
    def __init__(self, replacements: Mapping[str, str]) -> None:
        self._replacements = dict(replacements)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self._replacements.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        _visit_argument_expressions_with_transformer(node.args, self)
        if node.returns is not None:
            node.returns = typing.cast("ast.expr", self.visit(node.returns))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(typing.cast("ast.FunctionDef", node))

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        _visit_argument_expressions_with_transformer(node.args, self)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        node.bases = [typing.cast("ast.expr", self.visit(base)) for base in node.bases]
        node.keywords = [
            typing.cast("ast.keyword", self.visit(keyword_arg))
            for keyword_arg in node.keywords
        ]
        return node


def _visit_argument_expressions_with_transformer(
    args: ast.arguments, transformer: ast.NodeTransformer
) -> None:
    args.defaults = [
        typing.cast("ast.expr", transformer.visit(default)) for default in args.defaults
    ]
    args.kw_defaults = [
        None if default is None else typing.cast("ast.expr", transformer.visit(default))
        for default in args.kw_defaults
    ]
    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *(arg for arg in (args.vararg, args.kwarg) if arg is not None),
    ):
        if arg.annotation is not None:
            arg.annotation = typing.cast("ast.expr", transformer.visit(arg.annotation))


def _replace_code_identifiers(code: str, replacements: Mapping[str, str]) -> str:
    module = ast.parse(code, mode="exec")
    replaced = typing.cast(
        "ast.Module",
        _IdentifierReplacer(replacements).visit(module),
    )
    return ast.unparse(ast.fix_missing_locations(replaced))


def _code_stores_name(code: str, name: str) -> bool:
    module = ast.parse(code, mode="exec")
    return any(_statement_store_count(stmt, name) > 0 for stmt in module.body)


def _script_codes_output_name(
    codes: Sequence[str],
    *,
    active_name: str,
    current_name: str | None,
) -> str | None:
    candidates = [active_name]
    for name in (current_name, "derived"):
        if name is not None and name not in candidates:
            candidates.append(name)
    for name in candidates:
        for code in codes:
            try:
                if _code_stores_name(code, name):
                    return name
            except SyntaxError:
                continue
    return current_name


def _simplify_display_code(code: str, *, inline_targets: set[str] | None = None) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    body = module.body
    if not body:
        return code

    changed = False

    def current_code() -> str:
        if not changed:
            return code
        return ast.unparse(ast.fix_missing_locations(module))

    def drop_unused_inline_targets() -> None:
        nonlocal changed
        if inline_targets is None:
            return

        idx = 0
        while idx < len(body) - 1:
            stmt = body[idx]
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                idx += 1
                continue
            target_expr = stmt.targets[0]
            if (
                isinstance(target_expr, ast.Name)
                and target_expr.id in inline_targets
                and not any(
                    _statement_load_count(later, target_expr.id) > 0
                    for later in body[idx + 1 :]
                )
            ):
                del body[idx]
                changed = True
                continue
            idx += 1

    drop_unused_inline_targets()

    for stmt in body:
        if not isinstance(stmt, (ast.Assign, ast.Expr, ast.Import, ast.ImportFrom)):
            return current_code()

    while True:
        for idx, stmt in enumerate(body[:-1]):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue

            target_expr = stmt.targets[0]
            if not isinstance(target_expr, ast.Name):
                continue
            target = target_expr.id
            if inline_targets is not None and target not in inline_targets:
                continue
            later_loads = [
                later_idx
                for later_idx, later in enumerate(body[idx + 1 :], start=idx + 1)
                if _statement_load_count(later, target) > 0
            ]
            if len(later_loads) != 1:
                continue

            next_idx = later_loads[0]
            next_stmt = body[next_idx]
            if not isinstance(next_stmt, ast.Assign):
                continue
            replacement_load_names = {
                node.id
                for node in ast.walk(stmt.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            if (
                len(next_stmt.targets) != 1
                or not isinstance(next_stmt.targets[0], ast.Name)
                or _statement_load_count(next_stmt, target) != 1
                or any(
                    _statement_store_count(intervening, target)
                    for intervening in body[idx + 1 : next_idx]
                )
                or any(
                    _statement_store_count(intervening, name)
                    for intervening in body[idx + 1 : next_idx]
                    for name in replacement_load_names
                )
            ):
                continue

            new_stmt = typing.cast(
                "ast.stmt",
                _NameReplacer(target, _clone_expr(stmt.value)).visit(
                    _clone_stmt(next_stmt)
                ),
            )
            body[next_idx] = ast.fix_missing_locations(new_stmt)
            del body[idx]
            changed = True
            break
        else:
            break

    drop_unused_inline_targets()

    if not changed:
        return code
    return ast.unparse(ast.fix_missing_locations(module))


def _code_uses_name(code: str, name: str) -> bool:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    return any(_statement_load_count(stmt, name) > 0 for stmt in module.body)


def rebase_default_replay_input(code: str, input_name: str) -> str:
    """Replace the generic ``data`` replay input in generated code.

    Manager clipboard actions use this when a concrete source is known, such as a
    watched variable, a load snippet target, or a user-provided variable name.
    """
    if not _code_uses_name(code, "data"):
        return code

    try:
        input_expr = ast.parse(input_name, mode="eval").body
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    rebased = typing.cast("ast.Module", _NameReplacer("data", input_expr).visit(module))
    rebased = ast.fix_missing_locations(rebased)
    return _simplify_display_code(
        ast.unparse(rebased),
        inline_targets={"derived"},
    )


def uses_default_replay_input(code: str) -> bool:
    """Return whether generated replay code refers to the generic ``data`` input."""
    return _code_uses_name(code, "data")


_OPERATION_TYPES: dict[str, type[ToolProvenanceOperation]] = {}


def _operation_is(operation: ToolProvenanceOperation, *operation_names: str) -> bool:
    return getattr(operation, "op", None) in operation_names


def _operation_type(op: str) -> type[ToolProvenanceOperation]:
    operation_type = _OPERATION_TYPES.get(op)
    if operation_type is None:
        raise RuntimeError(
            "Concrete provenance operations are not registered. "
            "Import erlab.interactive.imagetool.provenance before "
            "constructing or deserializing operation-backed provenance."
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


def _receiver_path(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    if isinstance(node, ast.Name):
        return node.id, ()
    if isinstance(node, ast.Attribute):
        resolved = _receiver_path(node.value)
        if resolved is None:
            return None
        receiver_name, path = resolved
        return receiver_name, (*path, node.attr)
    return None


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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        """Apply this operation to the current derived array.

        Subclasses that participate in live refresh or executable provenance replay
        should reimplement this method. The implementation must be deterministic for
        the operation's stored model fields and must not mutate ``data`` or
        ``parent_data`` in place.

        Parameters
        ----------
        data
            Array produced by the preceding replay step.
        parent_data
            Parent ImageTool data for the enclosing provenance spec. Operations may
            inspect it for coordinates, dimension order, or metadata.

        Returns
        -------
        xarray.DataArray
            Array after this operation has been applied.

        Notes
        -----
        Operations that only emit generated code should set ``live_applicable = False``
        and raise from this method.
        """
        raise NotImplementedError

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

    def replay_code(
        self,
        input_name: str,
        *,
        output_name: str | None = None,
        source_name: str | None = None,
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
            return self.statement_code(
                input_name,
                output_name=output_name,
                source_name=source_name,
            )
        if output_name is None:
            return expression
        return f"{output_name} = {expression}"

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


def _expression_receiver_code(expression: str) -> str:
    """Return ``expression`` in a form that can safely receive another operation."""
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"({expression})"
    if isinstance(
        parsed.body,
        ast.Name | ast.Attribute | ast.Subscript | ast.Call,
    ):
        return expression
    return f"({expression})"


def operations_expression_code(
    operations: Sequence[ToolProvenanceOperation],
    input_name: str,
    *,
    source_name: str | None = None,
) -> str:
    """Return chained expression code for structured operations.

    ``source_name`` is the public/source array name passed to operations that need
    context beyond the transformed input, such as coordinate-order restoration.
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

    operation_type = _OPERATION_TYPES.get(op)
    if operation_type is None:
        raise ValueError(
            f"Unknown provenance operation {op!r}; "
            f"expected one of {sorted(_OPERATION_TYPES)}"
        )
    return operation_type.model_validate(value)


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
    return spec.model_copy(update={"operations": operations})


class FileDataSelection(pydantic.BaseModel):
    """Serializable selection of one displayable array from a loaded file object.

    New provenance stores stable Dataset variable names and DataTree data paths instead
    of the positional parsed-array index used by older workspaces.
    """

    kind: typing.Literal[
        "dataarray",
        "dataset_variable",
        "datatree_path",
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
    def _validate_serialized_shape(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, np.integer):
            return {"kind": "parsed_index", "value": int(value)}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"kind": "parsed_index", "value": value}
        return value

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
        if self.kind == "datatree_path":
            if not isinstance(self.value, str) or not self.value.startswith("/"):
                raise ValueError("datatree file selections must use an absolute path")
            return self

        value = int(self.value) if isinstance(self.value, np.integer) else self.value
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("parsed file selection index must be non-negative")
        if value != self.value:
            return self.model_copy(update={"value": value})
        return self

    @pydantic.field_serializer("value", when_used="json")
    def _serialize_value(self, value: typing.Any) -> typing.Any:
        if self.kind == "dataset_variable":
            return _encode_provenance_hashable(value)
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
    """Structured transformation stage replayed against one parent data array."""

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
        return tuple(
            parse_tool_provenance_operation(
                typing.cast("ToolProvenanceOperation | Mapping[str, typing.Any]", item)
            )
            for item in value
        )

    @pydantic.model_validator(mode="after")
    def _validate_live_operations(self) -> typing.Self:
        if any(not operation.live_applicable for operation in self.operations):
            raise TypeError("file replay stages cannot contain script-only operations")
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

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
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
        if self.source_kind == "full_data":
            return f"{input_name}.copy(deep=False)"
        return (
            f"erlab.interactive.imagetool.slicer.restore_nonuniform_dims({input_name})"
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


class ScriptInput(pydantic.BaseModel):
    """Named input captured by script or multi-tool provenance.

    ``name`` is the immutable replay variable, ``label`` is the historical display
    label, ``node_uid`` and ``node_snapshot_token`` identify the live manager input
    that was used, and ``provenance_spec`` stores the historical replay source used
    when that live input is unavailable.
    """

    name: str
    label: str
    node_uid: str | None = None
    node_snapshot_token: str | None = None
    provenance_spec: dict[str, typing.Any] | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

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
    origin: ``file_load_source`` is the file-origin capability, and
    ``replay_stages`` are structured operations that can be carried by both ``file`` and
    ``script`` replay specs. ``script`` specs may also include multi-input
    ``script_inputs`` for console or UI actions that combine several ImageTools.
    Deserialize saved payloads with
    :func:`parse_tool_provenance_spec`.

    A spec records exact operation arguments for live refresh, runtime replay, copied
    code, and derivation display. Manager children opened from ImageTool cursor or bin
    selections refresh by replaying those explicit arguments; legacy selection
    bindings are converted to source specs once for compatibility.
    """

    schema_version: typing.Literal[2] = 2
    kind: typing.Literal["full_data", "public_data", "selection", "script", "file"]
    start_label: str | None = None
    seed_code: str | None = None
    active_name: str | None = None
    operations: tuple[pydantic.SerializeAsAny[ToolProvenanceOperation], ...] = ()
    file_load_source: FileLoadSource | None = None
    replay_stages: tuple[ReplayStage, ...] = ()
    script_inputs: tuple[ScriptInput, ...] = ()
    script_context_bindings: tuple[_ScriptContextBinding, ...] = pydantic.Field(
        default=(),
        exclude_if=lambda value: not value,
    )

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_serialized_shape(cls, value: typing.Any) -> typing.Any:
        if (
            isinstance(value, Mapping)
            and value.get("kind") == "script"
            and "active_name" not in value
        ):
            raise ValueError("script provenance specs must define `active_name`")
        return value

    @pydantic.field_validator("active_name", mode="before")
    @classmethod
    def _validate_active_name_field(cls, value: typing.Any) -> str | None:
        return _validate_active_name(value)

    @pydantic.field_validator("operations", mode="before")
    @classmethod
    def _validate_operations(
        cls, value: typing.Any
    ) -> tuple[ToolProvenanceOperation, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized provenance operations must be a sequence")
        return tuple(
            parse_tool_provenance_operation(
                typing.cast("ToolProvenanceOperation | Mapping[str, typing.Any]", item)
            )
            for item in value
        )

    @pydantic.field_validator("replay_stages", mode="before")
    @classmethod
    def _validate_replay_stages(cls, value: typing.Any) -> tuple[ReplayStage, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized replay stages must be a sequence")
        return tuple(
            item if isinstance(item, ReplayStage) else ReplayStage.model_validate(item)
            for item in value
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

    @pydantic.field_validator("script_context_bindings", mode="before")
    @classmethod
    def _validate_script_context_bindings(
        cls, value: typing.Any
    ) -> tuple[_ScriptContextBinding, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("Serialized script context bindings must be a sequence")
        return tuple(
            item
            if isinstance(item, _ScriptContextBinding)
            else _ScriptContextBinding.model_validate(item)
            for item in value
        )

    @pydantic.model_validator(mode="after")
    def _validate_kind_fields(self) -> typing.Self:
        if self.kind == "script":
            if self.start_label is None:
                raise ValueError("script provenance specs must define `start_label`")
            for binding in self.script_context_bindings:
                if binding.operation_index >= len(self.operations):
                    raise ValueError(
                        "script context binding must target an operation boundary"
                    )
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
            if self.operations:
                raise ValueError("file provenance specs cannot define operations")
            if self.script_context_bindings:
                raise ValueError("file provenance specs cannot define script context")
            return self
        if (
            self.start_label is not None
            or self.seed_code is not None
            or self.active_name is not None
            or self.file_load_source is not None
            or self.replay_stages
            or self.script_inputs
            or self.script_context_bindings
        ):
            raise ValueError(
                "Only script or file provenance specs may define `start_label`, "
                "`seed_code`, `active_name`, `file_load_source`, `replay_stages`, "
                "`script_inputs`, or `script_context_bindings`"
            )
        return self

    @property
    def is_live_source(self) -> bool:
        return self.kind in {"full_data", "public_data", "selection"} and all(
            operation.live_applicable for operation in self.operations
        )

    def append_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Append operation instances to the spec.

        Runtime code should pass operation instances from this module. Saved mappings
        should be normalized with :func:`parse_tool_provenance_spec` before calling
        this method.
        """
        appended = tuple(_require_operation_instance(op) for op in operations)
        return self.model_copy(update={"operations": (*self.operations, *appended)})

    def drop_trailing_rename(self) -> ToolProvenanceSpec:
        operations = self.operations
        if operations and _operation_is(operations[-1], "rename"):
            operations = operations[:-1]
        return self.model_copy(update={"operations": operations})

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
                update={"operations": (*operations[:-1], operation, operations[-1])}
            )
        return self.append_operations(operation)

    def append_final_rename(self, name: str) -> ToolProvenanceSpec:
        return self.drop_trailing_rename().append_operations(
            _operation_instance("rename", name=name)
        )

    def append_replay_stage(self, source: ToolProvenanceSpec) -> ToolProvenanceSpec:
        """Append one live-source transformation stage to replay provenance."""
        if self.kind not in {"file", "script"}:
            raise TypeError("Replay stages can only be appended to replay provenance")
        stage = ReplayStage.from_source_spec(source)
        return self.model_copy(update={"replay_stages": (*self.replay_stages, stage)})

    def _operation_for_ref(
        self, ref: _ProvenanceStepRef
    ) -> ToolProvenanceOperation | None:
        if ref.kind != "operation" or ref.operation_index is None:
            return None
        if ref.stage_index is None:
            if 0 <= ref.operation_index < len(self.operations):
                return self.operations[ref.operation_index]
            return None
        if 0 <= ref.stage_index < len(self.replay_stages):
            operations = self.replay_stages[ref.stage_index].operations
            if 0 <= ref.operation_index < len(operations):
                return operations[ref.operation_index]
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

    def _operation_group_range_ref(
        self,
        ref: _ProvenanceStepRef,
        *,
        kind: str | None = None,
    ) -> tuple[int, int] | None:
        if ref.kind != "operation" or ref.operation_index is None:
            return None
        if ref.stage_index is None:
            operations = self.operations
        elif 0 <= ref.stage_index < len(self.replay_stages):
            operations = self.replay_stages[ref.stage_index].operations
        else:
            return None
        return operation_group_range(operations, ref.operation_index, kind=kind)

    def _replace_operation_group_ref(
        self,
        ref: _ProvenanceStepRef,
        replacements: Sequence[ToolProvenanceOperation],
        *,
        kind: str | None = None,
    ) -> ToolProvenanceSpec:
        group = self._operation_group_range_ref(ref, kind=kind)
        if group is None:
            raise ValueError("Expected a complete operation group reference")
        return self._replace_operation_range_ref(ref, group[0], group[1], replacements)

    def _delete_operation_group_ref(
        self,
        ref: _ProvenanceStepRef,
        *,
        kind: str | None = None,
    ) -> ToolProvenanceSpec:
        return self._replace_operation_group_ref(ref, (), kind=kind)

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
        if ref.stage_index is None:
            operations = list(self.operations)
            operations[start:stop] = replacement_ops
            updates: dict[str, typing.Any] = {"operations": tuple(operations)}
            if self.kind == "script" and self.script_context_bindings:
                operation_count_delta = len(replacement_ops) - (stop - start)
                script_context_bindings: list[_ScriptContextBinding] = []
                for binding in self.script_context_bindings:
                    operation_index = binding.operation_index
                    if start <= operation_index < stop and not (
                        operation_index == start and replacement_ops
                    ):
                        continue
                    if operation_index >= stop:
                        operation_index += operation_count_delta
                    if operation_index < len(operations):
                        script_context_bindings.append(
                            binding.model_copy(
                                update={"operation_index": operation_index}
                            )
                        )
                updates["script_context_bindings"] = tuple(script_context_bindings)
            return self.model_copy(update=updates)

        stages = list(self.replay_stages)
        stage = stages[ref.stage_index]
        operations = list(stage.operations)
        operations[start:stop] = replacement_ops
        stages[ref.stage_index] = stage.model_copy(
            update={"operations": tuple(operations)}
        )
        return self.model_copy(update={"replay_stages": tuple(stages)})

    def _prefix_through_ref(self, ref: _ProvenanceStepRef) -> ToolProvenanceSpec:
        """Return a spec replaying through the referenced displayed row."""
        if ref.kind in {"start", "file_load"}:
            if self.kind == "file":
                return self.model_copy(update={"replay_stages": ()})
            if self.kind in {"full_data", "public_data", "selection"}:
                return self.model_copy(update={"operations": ()})
            start_updates: dict[str, typing.Any] = {
                "operations": (),
                "replay_stages": (),
                "script_context_bindings": (),
            }
            if output_name := self._script_seed_output_name():
                start_updates["active_name"] = output_name
            return self.model_copy(update=start_updates)

        if ref.kind != "operation" or ref.operation_index is None:
            raise ValueError("This provenance row cannot be replayed as a prefix")

        end = ref.operation_index + 1
        if ref.stage_index is None:
            operation_updates: dict[str, typing.Any] = {
                "operations": self.operations[:end]
            }
            if self.kind == "script" and self.script_context_bindings:
                operation_updates["script_context_bindings"] = tuple(
                    binding
                    for binding in self.script_context_bindings
                    if binding.operation_index < end
                )
            return self.model_copy(update=operation_updates)

        stages = list(self.replay_stages[: ref.stage_index])
        stage = self.replay_stages[ref.stage_index]
        stages.append(stage.model_copy(update={"operations": stage.operations[:end]}))
        stage_updates: dict[str, typing.Any] = {"replay_stages": tuple(stages)}
        if self.kind == "script":
            stage_updates.update(
                {
                    "operations": (),
                    "script_context_bindings": (),
                }
            )
            if output_name := self._script_seed_output_name():
                stage_updates["active_name"] = output_name
        return self.model_copy(update=stage_updates)

    def _prefix_before_ref(self, ref: _ProvenanceStepRef) -> ToolProvenanceSpec:
        """Return a spec replaying to the input of the referenced operation row."""
        if ref.kind != "operation" or ref.operation_index is None:
            return self._prefix_through_ref(ref)

        if ref.stage_index is None:
            before_operation_updates: dict[str, typing.Any] = {
                "operations": self.operations[: ref.operation_index]
            }
            if self.kind == "script" and self.script_context_bindings:
                before_operation_updates["script_context_bindings"] = tuple(
                    binding
                    for binding in self.script_context_bindings
                    if binding.operation_index < ref.operation_index
                )
            return self.model_copy(update=before_operation_updates)

        stages = list(self.replay_stages[: ref.stage_index])
        stage = self.replay_stages[ref.stage_index]
        stages.append(
            stage.model_copy(
                update={"operations": stage.operations[: ref.operation_index]}
            )
        )
        before_stage_updates: dict[str, typing.Any] = {"replay_stages": tuple(stages)}
        if self.kind == "script":
            before_stage_updates.update(
                {
                    "operations": (),
                    "script_context_bindings": (),
                }
            )
            if output_name := self._script_seed_output_name():
                before_stage_updates["active_name"] = output_name
        return self.model_copy(update=before_stage_updates)

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
                    (
                        not getattr(operation, "visible", True)
                        and not include_hidden_script_code
                    )
                    or entry.code
                    in {
                        "derived = derived.isel()",
                        "derived = derived.qsel()",
                        "derived = derived.sel()",
                    }
                    or _is_internal_sort_coord_order_entry(entry)
                )
            # Rule 1: drop empty selection operations.
            elif _operation_is(operation, "qsel", "isel", "sel"):
                hide_operation = not getattr(operation, "decoded_kwargs", {})
            # Rule 2: hide internal coordinate-order normalization.
            elif _operation_is(operation, "sort_coord_order", "source_view"):
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
            operations=self.operations,
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
            data = operation.apply(data, parent_data=parent_data)
        return data

    def derivation_entries(self) -> list[DerivationEntry]:
        entries: list[DerivationEntry] = [self._start_entry()]
        if self.kind == "script":
            entries.extend(
                DerivationEntry(
                    f"Use {script_input.name} from {script_input.label}",
                    None,
                    False,
                )
                for script_input in self.script_inputs
            )
        operations: Sequence[ToolProvenanceOperation]
        if self.kind == "file":
            operations = tuple(
                operation
                for stage in self.replay_stages
                for operation in stage.operations
            )
        elif self.kind == "script":
            operations = (
                *(
                    operation
                    for stage in self.replay_stages
                    for operation in stage.operations
                ),
                *self.operations,
            )
        else:
            operations = self.operations
        for operation in operations:
            entry = operation.derivation_entry()
            entries.append(entry)
        return entries

    def _script_graph_code(self, *, display: bool) -> str | None:
        if not self.operations and not self.replay_stages:
            return None
        try:
            graph = _replay_graph.compile_replay_graph(self, display=display)
            return _replay_graph.emit_replay_code(
                graph,
                output_name=typing.cast("str", self.active_name),
            )
        except _replay_graph.ReplayGraphError:
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
                f"Use {script_input.name} from {script_input.label}",
                None,
                False,
            )
            for script_input in self.script_inputs
        )

        stage_parent_data = parent_data
        for stage in self.replay_stages:
            entries.extend(
                operation.derivation_entry()
                for _, operation in self._streamlined_operation_refs(
                    stage.source_kind,
                    stage.operations,
                    parent_data=stage_parent_data,
                )
            )
            stage_parent_data = None

        entries.extend(
            operation.derivation_entry()
            for _, operation in self._streamlined_operation_refs(
                "full_data",
                self.operations,
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
        return "\n".join(part for part in (prefix, *step_codes) if part)

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
                            f"Use {script_input.name} from {script_input.label}",
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
            stage_parent_data = parent_data
            for stage_index, stage in enumerate(self.replay_stages):
                for operation_index, operation in self._streamlined_operation_refs(
                    stage.source_kind,
                    stage.operations,
                    parent_data=stage_parent_data,
                ):
                    step_ref = _ProvenanceStepRef(
                        "operation",
                        operation_index=operation_index,
                        stage_index=stage_index,
                    )
                    rows.append(
                        _ProvenanceDisplayRow(
                            operation.derivation_entry(),
                            edit_ref=step_ref,
                            replay_ref=step_ref,
                        )
                    )
                stage_parent_data = None
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
            stage_parent_data = parent_data
            for stage_index, stage in enumerate(self.replay_stages):
                for operation_index, operation in self._streamlined_operation_refs(
                    stage.source_kind,
                    stage.operations,
                    parent_data=stage_parent_data,
                ):
                    step_ref = _ProvenanceStepRef(
                        "operation",
                        operation_index=operation_index,
                        stage_index=stage_index,
                    )
                    rows.append(
                        _ProvenanceDisplayRow(
                            operation.derivation_entry(),
                            edit_ref=step_ref,
                            replay_ref=step_ref,
                        )
                    )
                stage_parent_data = None
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
            inline_targets = {"derived"} if self.active_name != "derived" else None
            return _simplify_display_code(
                graph_code,
                inline_targets=inline_targets,
            )
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
    if value.get("schema_version") == 1:
        value = dict(value)
        value["schema_version"] = 2
    return _normalize_script_code_operations(ToolProvenanceSpec.model_validate(value))


def has_file_load_source(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> bool:
    """Return whether a spec records a file origin, independent of replay kind."""
    spec = parse_tool_provenance_spec(value)
    return spec is not None and spec.file_load_source is not None


def file_load_source_status(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> FileLoadSourceStatus:
    """Return the current availability of the recorded file-load source."""
    spec = parse_tool_provenance_spec(value)
    if spec is None or spec.file_load_source is None:
        return "no-file-load-source"
    load_source = spec.file_load_source
    if not pathlib.Path(load_source.path).exists():
        return "missing-file"
    replay_call = load_source.replay_call
    if replay_call is None:
        return "no-replay-call"
    if (
        replay_call.kind == "erlab_loader"
        and replay_call.target not in erlab.io.loaders
    ):
        return "missing-loader"
    if replay_call.kind == "callable":
        try:
            _resolve_importable_callable(replay_call.target)
        except (AttributeError, ModuleNotFoundError, TypeError, ValueError):
            return "missing-loader"
    return "loadable"


def iter_operation_refs(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> typing.Iterator[tuple[_ProvenanceStepRef, ToolProvenanceOperation]]:
    """Yield operation references in replay order across all operation stores."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return
    if spec.kind in {"file", "script"}:
        for stage_index, stage in enumerate(spec.replay_stages):
            for operation_index, operation in enumerate(stage.operations):
                yield (
                    _ProvenanceStepRef(
                        "operation",
                        operation_index=operation_index,
                        stage_index=stage_index,
                    ),
                    operation,
                )
    if spec.kind in {"full_data", "public_data", "selection", "script"}:
        for operation_index, operation in enumerate(spec.operations):
            yield (
                _ProvenanceStepRef("operation", operation_index=operation_index),
                operation,
            )


def can_reload_without_trust(
    value: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> bool:
    """Return whether recorded provenance can replay without trusted user code."""
    spec = parse_tool_provenance_spec(value)
    if spec is None:
        return False
    if spec.kind == "file":
        return file_load_source_status(spec) == "loadable"
    if spec.kind != "script":
        return False
    if has_file_load_source(spec) and file_load_source_status(spec) != "loadable":
        return False
    if not script_provenance_replayable(spec):
        return False
    for script_input in spec.script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if not can_reload_without_trust(input_spec):
            return False
    return True


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
            if script_input.node_uid is not None:
                refs.append(
                    ScriptInputDependencyRef(
                        name=script_input.name,
                        label=script_input.label,
                        node_uid=script_input.node_uid,
                        node_snapshot_token=script_input.node_snapshot_token,
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
            if _operation_is(operation, "sort_coord_order"):
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
        seed_code=_default_seed_code_for_operations(display_operations),
        active_name="derived",
        operations=display_operations,
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
        or spec.replay_stages
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
            replay_stages=spec.replay_stages,
        )
    return spec.to_replay_spec()


def _script_operations_from_replay_stages(
    replay_stages: Sequence[ReplayStage],
) -> tuple[ToolProvenanceOperation, ...]:
    operations: list[ToolProvenanceOperation] = []
    for stage in replay_stages:
        if stage.source_kind != "full_data":
            operations.append(_SourceViewOperation(source_kind=stage.source_kind))
        operations.extend(stage.operations)
    return tuple(operations)


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
    if parent_replay.kind == "file":
        with contextlib.suppress(TypeError):
            local_live = require_live_source_spec(local_value)
            if local_live is not None:
                if local_live.kind in {"full_data", "public_data"} and not (
                    local_live.operations
                ):
                    return parent_replay
                return parent_replay.append_replay_stage(local_live)

    parent_spec = _as_script_replay_spec(parent_replay)
    local_spec = _as_script_replay_spec(local_value.to_replay_spec())

    normalized_context_names: list[str] = []
    for item in script_context_names:
        name = _validate_active_name(item)
        if name is None:
            raise ValueError("script context names must not be None")
        if name not in normalized_context_names:
            normalized_context_names.append(name)

    local_stage_operations = _script_operations_from_replay_stages(
        local_spec.replay_stages
    )
    local_operations = [*local_stage_operations, *local_spec.operations]
    local_operation_offset = len(parent_spec.operations)
    inserted_local_operations = 0
    if local_spec.seed_code:
        seed_code: str | None = local_spec.seed_code
        if seed_code == _DEFAULT_REPLAY_SEED_CODE:
            parent_input = replay_input_name(parent_spec)
            if parent_input == "derived" or parent_spec.active_name == "derived":
                seed_code = None
            elif parent_input is not None:
                seed_code = f"derived = {parent_input}"
        if seed_code is not None:
            local_operations.insert(
                0,
                _operation_instance(
                    "script_code",
                    label=typing.cast("str", local_spec.start_label),
                    code=seed_code,
                    visible=False,
                ),
            )
            inserted_local_operations = 1
    elif local_spec.active_name == "derived":
        parent_input = replay_input_name(parent_spec)
        if (
            parent_input is not None
            and parent_input != "derived"
            and parent_spec.active_name not in {None, "derived"}
            and local_operations
        ):
            local_operations.insert(
                0,
                _operation_instance(
                    "script_code",
                    label="Use current parent output as the active data",
                    code=f"derived = {parent_input}",
                    visible=False,
                ),
            )
            inserted_local_operations = 1

    script_context_bindings = list(parent_spec.script_context_bindings)
    if normalized_context_names and local_value.kind == "script" and local_operations:
        script_context_bindings.append(
            _ScriptContextBinding(
                operation_index=local_operation_offset,
                names=tuple(normalized_context_names),
            )
        )
    script_context_bindings.extend(
        binding.model_copy(
            update={
                "operation_index": (
                    local_operation_offset
                    + inserted_local_operations
                    + len(local_stage_operations)
                    + binding.operation_index
                )
            }
        )
        for binding in local_spec.script_context_bindings
    )

    return ToolProvenanceSpec(
        kind="script",
        start_label=typing.cast("str", parent_spec.start_label),
        seed_code=parent_spec.seed_code,
        active_name=local_spec.active_name or parent_spec.active_name,
        file_load_source=parent_spec.file_load_source or local_spec.file_load_source,
        replay_stages=parent_spec.replay_stages,
        script_inputs=(*parent_spec.script_inputs, *local_spec.script_inputs),
        operations=(*parent_spec.operations, *local_operations),
        script_context_bindings=tuple(script_context_bindings),
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
    replay_stages: Sequence[ReplayStage] = (),
    script_inputs: Sequence[ScriptInput] = (),
) -> ToolProvenanceSpec:
    """Build script provenance from code, structured steps, and named inputs."""
    return ToolProvenanceSpec(
        kind="script",
        start_label=start_label,
        seed_code=seed_code,
        active_name=active_name,
        file_load_source=file_load_source,
        replay_stages=tuple(replay_stages),
        script_inputs=tuple(script_inputs),
    ).append_operations(*operations)


def file_load(
    *,
    start_label: str,
    seed_code: str,
    file_load_source: FileLoadSource,
    active_name: str = "derived",
    replay_stages: Sequence[ReplayStage] = (),
) -> ToolProvenanceSpec:
    """Build structured file-backed provenance for runtime reload."""
    return ToolProvenanceSpec(
        kind="file",
        start_label=start_label,
        seed_code=seed_code,
        active_name=active_name,
        file_load_source=file_load_source,
        replay_stages=tuple(replay_stages),
    )


def _processed_replay_ndim(darr: xr.DataArray) -> int:
    if darr.ndim == 1:
        return 2
    if darr.ndim > 4:
        return len(tuple(size for size in darr.shape if size != 1))
    return darr.ndim


def _supported_replay_shape(darr: xr.DataArray) -> bool:
    return _processed_replay_ndim(darr) in (2, 3, 4)


def _reducible_replay_shape(darr: xr.DataArray) -> bool:
    return _processed_replay_ndim(darr) >= 2


def _parse_replay_dataset(ds: xr.Dataset) -> tuple[xr.DataArray, ...]:
    return tuple(
        darr for darr in ds.data_vars.values() if _reducible_replay_shape(darr)
    )


def _parse_replay_input(data: typing.Any) -> list[xr.DataArray]:
    input_cls = data.__class__.__name__
    parsed: typing.Any = data
    if isinstance(data, np.ndarray | xr.DataArray):
        parsed = (data,)
    elif isinstance(data, xr.Dataset):
        parsed = _parse_replay_dataset(data)
    elif isinstance(data, xr.DataTree):
        parsed = tuple(
            darr for leaf in data.leaves for darr in _parse_replay_dataset(leaf.dataset)
        )

    if len(parsed) == 0:
        raise ValueError(f"No valid data for ImageTool found in {input_cls}")
    if not isinstance(next(iter(parsed)), xr.DataArray | np.ndarray):
        raise TypeError(
            f"Unsupported input type {input_cls}. Expected DataArray, Dataset, "
            "DataTree, numpy array, or a list of DataArray or numpy arrays."
        )
    return [
        xr.DataArray(item) if not isinstance(item, xr.DataArray) else item
        for item in parsed
    ]


def _require_replay_dataarray(data: typing.Any) -> xr.DataArray:
    if isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Selected file data must be a DataArray, got {type(data).__name__!r}"
        )
    if not _reducible_replay_shape(data):
        raise ValueError("Selected file data is not valid for ImageTool")
    return data


def _select_replay_input(
    data: typing.Any,
    selection: FileDataSelection | Mapping[str, typing.Any] | int,
) -> xr.DataArray:
    selection = FileDataSelection.model_validate(selection)
    if selection.kind == "dataarray":
        return _require_replay_dataarray(data)
    if selection.kind == "dataset_variable":
        if not isinstance(data, xr.Dataset):
            raise TypeError(
                "Dataset variable file selections require the loader to return "
                "a Dataset"
            )
        try:
            selected = data[selection.value]
        except KeyError as err:
            raise KeyError(
                f"Selected file variable {selection.value!r} was not found"
            ) from err
        return _require_replay_dataarray(selected)
    if selection.kind == "datatree_path":
        if not isinstance(data, xr.DataTree):
            raise TypeError(
                "DataTree path file selections require the loader to return a DataTree"
            )
        try:
            selected = data[selection.value]
        except KeyError as err:
            raise KeyError(
                f"Selected file DataTree path {selection.value!r} was not found"
            ) from err
        return _require_replay_dataarray(selected)

    parsed = _parse_replay_input(data)
    index = typing.cast("int", selection.value)
    if index >= len(parsed):
        raise IndexError(
            f"Selected file replay index {index} is out of range for "
            f"{len(parsed)} parsed arrays"
        )
    return parsed[index]


def _resolve_importable_callable(target: str) -> Callable[..., typing.Any]:
    parts = target.split(".")
    if len(parts) < 2:
        raise ValueError(f"Importable callable target {target!r} must be dotted")

    module = None
    attr_start = 0
    for idx in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        attr_start = idx
        break
    if module is None:
        raise ModuleNotFoundError(target)

    obj: typing.Any = module
    for attr in parts[attr_start:]:
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"Importable target {target!r} is not callable")
    return typing.cast("Callable[..., typing.Any]", obj)


def _load_file_source_data(load_source: FileLoadSource) -> xr.DataArray:
    call = load_source.replay_call
    if call is None:
        raise ValueError("File load source does not define replay metadata")
    file_path = pathlib.Path(load_source.path)
    if call.kind == "erlab_loader":
        func = erlab.io.loaders[call.target].load
    else:
        func = _resolve_importable_callable(call.target)

    loaded = func(file_path, **dict(call.kwargs))
    data = _select_replay_input(loaded, call.selection)
    if call.cast_float64:
        data = data.astype(np.float64)
    return data


def _apply_replay_stage(stage: ReplayStage, parent_data: xr.DataArray) -> xr.DataArray:
    data = ToolProvenanceSpec._starting_data_for_kind(stage.source_kind, parent_data)
    for operation in stage.operations:
        data = operation.apply(data, parent_data=parent_data)
    return data


def replay_file_provenance(
    spec: ToolProvenanceSpec | Mapping[str, typing.Any],
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    """Replay structured file provenance without executing generated Python."""
    try:
        return _replay_graph.replay_file_provenance(spec, cache=cache)
    except _replay_graph.ReplayGraphError as exc:
        raise TypeError("Expected structured file provenance") from exc


def _validate_script_replay_code(code: str) -> None:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Script replay code is not valid Python: {exc}") from exc

    def validate_import(node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
        else:
            if node.level != 0 or node.module is None:
                raise TypeError("Script replay code contains unsupported ImportFrom")
            roots = {node.module.partition(".")[0]}
        if not roots <= _SCRIPT_REPLAY_ALLOWED_IMPORT_ROOTS:
            raise TypeError(
                f"Script replay code contains unsupported {type(node).__name__}"
            )

    def validate_optional_import_try(node: ast.Try) -> None:
        if (
            node.finalbody
            or len(node.handlers) != 1
            or not node.body
            or any(
                not isinstance(statement, ast.Import | ast.ImportFrom)
                for statement in node.body
            )
        ):
            raise TypeError("Script replay code contains unsupported Try")
        handler = node.handlers[0]
        if (
            handler.name is not None
            or not isinstance(handler.type, ast.Name)
            or handler.type.id != "ImportError"
            or len(handler.body) != 1
            or not isinstance(handler.body[0], ast.Pass)
        ):
            raise TypeError("Script replay code contains unsupported Try")

    for node in ast.walk(module):
        if isinstance(node, ast.Import | ast.ImportFrom):
            validate_import(node)
            continue
        if isinstance(node, ast.Try):
            validate_optional_import_try(node)
            continue
        if isinstance(node, _SCRIPT_REPLAY_FORBIDDEN_NODES):
            raise TypeError(
                f"Script replay code contains unsupported {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Script replay code cannot access dunder names")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Script replay code cannot access dunder attributes")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _SCRIPT_REPLAY_FORBIDDEN_CALLS
        ):
            raise ValueError(f"Script replay code cannot call {node.func.id!r}")


def script_provenance_replayable(
    spec: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
) -> bool:
    """Return whether script provenance is self-contained enough to execute.

    This checks the saved code and structured operations. It does not mean live
    manager inputs are present; callers still need to resolve ``script_inputs``.
    """
    return _replay_graph.script_provenance_replayable(spec)


def script_provenance_requires_trust(
    spec: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    *,
    external_input_names: set[str] | None = None,
) -> bool:
    """Return whether script provenance can replay only after user trust."""
    return _replay_graph.script_provenance_requires_trust(
        spec,
        external_input_names=external_input_names,
    )


def replay_script_provenance(
    spec: ToolProvenanceSpec | Mapping[str, typing.Any],
    inputs: Mapping[str, xr.DataArray],
    *,
    trusted_user_code: bool = False,
) -> xr.DataArray:
    """Execute script provenance from already resolved input arrays.

    The caller is responsible for trust and input resolution. This function only
    validates the provenance shape, compiles it through the replay graph, and returns
    the replayed :class:`xarray.DataArray`.
    """
    try:
        return _replay_graph.replay_script_provenance(
            spec,
            inputs,
            trusted_user_code=trusted_user_code,
        )
    except _replay_graph.ReplayGraphError as exc:
        if "non-replayable" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(str(exc)) from exc
