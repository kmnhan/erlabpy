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

Manager children opened from an ImageTool cursor or bin selection do not keep the
first generated ``qsel`` arguments as their refresh state. They store the selected
parent indices in :class:`ImageToolSelectionSourceBinding` and rebuild ``qsel`` or
``isel`` operations from the current parent data each time they refresh.

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

5. Implement :meth:`ToolProvenanceOperation.derivation_entry` so the manager can display
   a user-facing summary and optional copyable code. Return a :class:`DerivationEntry`
   for every concrete operation; use ``code=None`` when the step should be visible but
   copied/replay code cannot be emitted safely.

6. If the operation maps cleanly from manager-console calls, declare
   :attr:`ToolProvenanceOperation.console_patterns` or implement
   :meth:`ToolProvenanceOperation.from_console_call`. Unsupported or ambiguous console
   calls should return ``None`` so they remain valid script provenance instead of being
   recorded as lossy structured operations.

7. Make generated code executable in the replay namespace. Use the literal helpers in
   this module for persisted values and make sure code assigns the ``derived`` replay
   variable; the replay graph may rebase ``data`` and ``derived`` to internal names.

8. Export the operation class from this module so runtime call sites can instantiate
   it directly.

9. Add tests that cover round-trip validation, :meth:`apply`, derivation text/code,
   console matching when supported, and any save/load or reload path that persists the
   new operation.

Parsing of serialized payloads happens only through :func:`parse_tool_provenance_spec`
and :func:`parse_tool_provenance_operation`. Runtime authoring code should create specs
with :func:`full_data`, :func:`public_data`, :func:`selection`, :func:`file_load`, or
:func:`script`, then instantiate operation models from this module directly.
"""

from __future__ import annotations

__all__ = [
    "AffineCoordOperation",
    "AssignAttrsOperation",
    "AssignCoord1DOperation",
    "AssignCoordsOperation",
    "AssignScalarCoordOperation",
    "AverageOperation",
    "CoarsenOperation",
    "CorrectWithEdgeOperation",
    "DerivationEntry",
    "DivideByCoordOperation",
    "FileLoadSource",
    "FileReplayCall",
    "ImageToolSelectionSourceBinding",
    "InterpolationOperation",
    "IselOperation",
    "LeadingEdgeOperation",
    "MaskWithPolygonOperation",
    "QSelAggregationOperation",
    "QSelOperation",
    "RenameDimsCoordsOperation",
    "RenameOperation",
    "ReplayStage",
    "RestoreNonuniformDimsOperation",
    "RotateOperation",
    "ScriptCodeOperation",
    "ScriptInput",
    "ScriptInputDependencyRef",
    "SelOperation",
    "SelectCoordOperation",
    "SliceAlongPathOperation",
    "SortCoordOrderOperation",
    "SqueezeOperation",
    "SwapDimsOperation",
    "SymmetrizeNfoldOperation",
    "SymmetrizeOperation",
    "ThinOperation",
    "ToolProvenanceOperation",
    "ToolProvenanceSpec",
    "TransposeOperation",
    "compose_display_provenance",
    "compose_full_provenance",
    "decode_provenance_value",
    "direct_replay_input_name",
    "encode_provenance_value",
    "file_load",
    "full_data",
    "mark_promoted_1d_source",
    "parse_tool_provenance_spec",
    "public_data",
    "rebase_default_replay_input",
    "rebase_script_input_node_uids",
    "replay_file_provenance",
    "replay_script_provenance",
    "require_live_source_spec",
    "script",
    "script_input_dependency_refs",
    "script_provenance_replayable",
    "selection",
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
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pydantic
import xarray as xr

import erlab
from erlab.interactive.imagetool import _replay_graph

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
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "slice": slice,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}
_SCRIPT_REPLAY_FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_SCRIPT_REPLAY_FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "open"}


@dataclass(frozen=True)
class DerivationEntry:
    """One user-visible step in a provenance derivation listing."""

    label: str
    code: str | None
    copyable: bool = False


@dataclass(frozen=True)
class ScriptInputDependencyRef:
    """Live manager dependency captured by a script input."""

    name: str
    label: str
    node_uid: str
    node_snapshot_token: str | None = None


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


def _format_selection_step(method: str, kwargs: Mapping[Hashable, typing.Any]) -> str:
    if not kwargs:
        return f"derived = derived.{method}()"
    args = erlab.interactive.utils.format_kwargs(dict(kwargs))
    return f"derived = derived.{method}({args})"


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


def _simplify_display_code(code: str, *, inline_targets: set[str] | None = None) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    body = module.body
    if not body:
        return code

    for stmt in body:
        if not isinstance(stmt, (ast.Assign, ast.Expr, ast.Import, ast.ImportFrom)):
            return code
        if isinstance(stmt, ast.Assign) and (
            len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name)
        ):
            return code

    changed = False
    while True:
        for idx, stmt in enumerate(body[:-1]):
            if not isinstance(stmt, ast.Assign):
                continue

            target_expr = stmt.targets[0]
            if not isinstance(target_expr, ast.Name):
                return code
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
            replacement_load_names = {
                node.id
                for node in ast.walk(stmt.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            if (
                not isinstance(next_stmt, (ast.Assign, ast.Expr))
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
    implement :meth:`apply` to replay the transformation, and implement
    :meth:`derivation_entry` to describe the step in manager UI.

    Operation instances store the exact arguments used by live refresh, replay graph
    execution, copied code, and derivation display. If a public console call maps
    exactly to an operation, expose it with ``console_patterns`` or
    :meth:`from_console_call`; ambiguous calls should return ``None`` so script
    provenance records the original code instead.
    """

    live_applicable: typing.ClassVar[bool] = True
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = ()

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

        Every concrete operation returns a :class:`DerivationEntry` so the manager can
        display it and copied provenance code has a single replay contract.

        Use ``DerivationEntry(..., code=None)`` instead when the step should remain
        visible in the derivation list but code generation should stop and return
        ``None``.
        """
        raise NotImplementedError

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        for pattern in cls.console_patterns:
            values = pattern.match(call)
            if values is None:
                continue
            with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
                return cls(**values)
        return None


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


class FileReplayCall(pydantic.BaseModel):
    """Serializable call information used to reload file-backed provenance."""

    kind: typing.Literal["erlab_loader", "callable"]
    target: str
    kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    selected_index: int
    cast_float64: bool = False

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="after")
    def _validate_replay_call(self) -> typing.Self:
        if self.selected_index < 0:
            raise ValueError("selected_index must be non-negative")
        if not self.target:
            raise ValueError("target must not be empty")
        if any(not isinstance(key, str) for key in self.kwargs):
            raise TypeError("file replay kwargs must use string keys")
        return self


class ReplayStage(pydantic.BaseModel):
    """Structured transformation stage replayed against one parent data array."""

    source_kind: typing.Literal["full_data", "public_data", "selection"]
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
            source_kind=typing.cast(
                "typing.Literal['full_data', 'public_data', 'selection']",
                live_source.kind,
            ),
            operations=live_source.operations,
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


class ToolProvenanceSpec(pydantic.BaseModel):
    """Saved provenance recipe for ImageTool data.

    Live child-tool refresh uses single-parent specs from :func:`full_data`,
    :func:`public_data`, or :func:`selection`. Durable reload and copied code use
    ``file`` and ``script`` specs, including multi-input ``script_inputs`` for console
    or UI actions that combine several ImageTools. Deserialize saved payloads with
    :func:`parse_tool_provenance_spec`.

    A spec records exact operation arguments for live refresh, runtime replay, copied
    code, and derivation display. Manager children opened from ImageTool cursor or bin
    selections should keep :class:`ImageToolSelectionSourceBinding` as their refresh
    state; that binding builds a spec before refresh so edited parent coordinates are
    used.

    .. versionchanged:: 3.23.0
       Script provenance can describe multiple manager inputs and records manager
       snapshot tokens so derived tools can show whether the live inputs still match
       the inputs used to create them.
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

    @pydantic.model_validator(mode="after")
    def _validate_kind_fields(self) -> typing.Self:
        if self.kind == "script":
            if self.start_label is None:
                raise ValueError("script provenance specs must define `start_label`")
            if self.replay_stages:
                raise ValueError("script provenance specs cannot define replay stages")
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
            return self
        if (
            self.start_label is not None
            or self.seed_code is not None
            or self.active_name is not None
            or self.file_load_source is not None
            or self.replay_stages
            or self.script_inputs
        ):
            raise ValueError(
                "Only script or file provenance specs may define `start_label`, "
                "`seed_code`, `active_name`, `file_load_source`, `replay_stages`, "
                "or `script_inputs`"
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
        if operations and isinstance(operations[-1], RenameOperation):
            operations = operations[:-1]
        return self.model_copy(update={"operations": operations})

    def append_replacement_operations(
        self, *operations: ToolProvenanceOperation
    ) -> ToolProvenanceSpec:
        """Replace a final rename, if present, then append new operation instances."""
        return self.drop_trailing_rename().append_operations(*operations)

    def append_final_rename(self, name: str) -> ToolProvenanceSpec:
        return self.drop_trailing_rename().append_operations(RenameOperation(name=name))

    def append_replay_stage(self, source: ToolProvenanceSpec) -> ToolProvenanceSpec:
        """Append one live-source transformation stage to file provenance."""
        if self.kind != "file":
            raise TypeError("Replay stages can only be appended to file provenance")
        stage = ReplayStage.from_source_spec(source)
        return self.model_copy(update={"replay_stages": (*self.replay_stages, stage)})

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
        source_kind: typing.Literal["full_data", "public_data", "selection"],
        parent_data: xr.DataArray,
    ) -> xr.DataArray:
        if source_kind == "full_data":
            return parent_data.copy(deep=False)
        return erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            parent_data.copy(deep=False)
        )

    @staticmethod
    def _streamlined_operations(
        source_kind: typing.Literal["full_data", "public_data", "selection"],
        operations: Sequence[ToolProvenanceOperation],
        *,
        parent_data: xr.DataArray | None = None,
    ) -> tuple[ToolProvenanceOperation, ...]:
        current_data: xr.DataArray | None = None
        if parent_data is not None:
            current_data = ToolProvenanceSpec._starting_data_for_kind(
                source_kind,
                parent_data,
            )

        streamlined: list[ToolProvenanceOperation] = []
        for index, operation in enumerate(operations):
            hide_operation = False

            if isinstance(operation, ScriptCodeOperation):
                entry = operation.derivation_entry()
                hide_operation = entry.code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                } or _is_internal_sort_coord_order_entry(entry)
                if not hide_operation and _is_whole_array_rename_entry(entry):
                    hide_operation = not any(
                        isinstance(later_operation, ScriptCodeOperation)
                        for later_operation in operations[index + 1 :]
                    )
            # Rule 1: drop empty selection operations.
            elif isinstance(operation, (QSelOperation, IselOperation, SelOperation)):
                hide_operation = not operation.decoded_kwargs
            # Rule 2: hide internal coordinate-order normalization.
            elif isinstance(operation, SortCoordOrderOperation):
                hide_operation = True
            # Rule 3: drop transpose calls that do not change dimension order.
            elif isinstance(operation, TransposeOperation) and current_data is not None:
                target_dims = (
                    tuple(operation.dims)
                    if operation.dims is not None
                    else tuple(reversed(current_data.dims))
                )
                hide_operation = target_dims == tuple(current_data.dims)
            # Rule 4: drop squeeze calls that would not remove singleton dimensions.
            elif isinstance(operation, SqueezeOperation) and current_data is not None:
                hide_operation = not any(size == 1 for size in current_data.shape)
            # Rule 5: drop nonuniform restoration when it would not change dimensions.
            elif (
                isinstance(operation, RestoreNonuniformDimsOperation)
                and current_data is not None
            ):
                hide_operation = (
                    erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                        current_data
                    ).dims
                    == current_data.dims
                )
            # Rule 6: drop whole-array name changes. ImageTool appends names such as
            # ``_avg`` to keep displayed tools distinct, but the DataArray name is not
            # an analysis step. Dimension and coordinate renames use
            # RenameDimsCoordsOperation and remain visible.
            elif isinstance(operation, RenameOperation):
                hide_operation = not any(
                    isinstance(later_operation, ScriptCodeOperation)
                    for later_operation in operations[index + 1 :]
                )

            if not hide_operation:
                streamlined.append(operation)

            # Rule 7: keep anything ambiguous. If replaying an operation fails while
            # building the heuristic context, stop making data-dependent decisions for
            # later steps and preserve them verbatim.
            if current_data is None or parent_data is None:
                current_data = None
            else:
                try:
                    current_data = operation.apply(
                        current_data, parent_data=parent_data
                    )
                except Exception:
                    current_data = None

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
        operations without ``exec``. Live ImageTool refresh uses the original
        single-parent spec via
        :func:`require_live_source_spec`.
        """
        if self.kind in {"script", "file"}:
            return self

        entries = self.derivation_entries()
        return ToolProvenanceSpec(
            kind="script",
            start_label=entries[0].label,
            seed_code=_DEFAULT_REPLAY_SEED_CODE,
            active_name="derived",
            file_load_source=self.file_load_source,
            operations=tuple(
                ScriptCodeOperation(
                    label=entry.label,
                    code=entry.code,
                    copyable=entry.copyable,
                )
                for entry in entries[1:]
            ),
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
        else:
            operations = self.operations
        for operation in operations:
            entry = operation.derivation_entry()
            entries.append(entry)
        return entries

    def _script_graph_code(self, *, display: bool) -> str | None:
        if not self.script_inputs:
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

    def derivation_code(self) -> str | None:
        prefix: str | None = None
        if self.kind == "script" and self.script_inputs:
            return self._script_graph_code(display=True)
        if self.kind in {"script", "file"}:
            prefix = self.seed_code
        step_codes = self._code_lines_from_entries(self.display_entries()[1:])
        if step_codes is None:
            return None
        if prefix is None and self.kind != "script":
            if not step_codes:
                return None
            prefix = _DEFAULT_REPLAY_SEED_CODE
        if prefix is None and not step_codes:
            return None
        return "\n".join(part for part in (prefix, *step_codes) if part)

    def display_entries(
        self, *, parent_data: xr.DataArray | None = None
    ) -> list[DerivationEntry]:
        """Return streamlined derivation entries for UI and copy-code output.

        The display path hides internal ImageTool normalization steps while keeping the
        recorded replay steps available through :meth:`derivation_entries`.
        """
        entries = [self._start_entry()]

        if self.kind == "script":
            entries.extend(
                DerivationEntry(
                    f"Use {script_input.name} from {script_input.label}",
                    None,
                    False,
                )
                for script_input in self.script_inputs
            )
            entries.extend(
                operation.derivation_entry()
                for operation in self._streamlined_operations(
                    "full_data",
                    self.operations,
                )
            )
            return entries
        if self.kind == "file":
            current_data = parent_data
            for stage in self.replay_stages:
                for operation in self._streamlined_operations(
                    stage.source_kind,
                    stage.operations,
                    parent_data=current_data,
                ):
                    operation_entry = operation.derivation_entry()
                    entries.append(operation_entry)
                if current_data is not None:
                    try:
                        current_data = _apply_replay_stage(stage, current_data)
                    except Exception:
                        current_data = None
            return entries

        entries.extend(
            operation.derivation_entry()
            for operation in self._display_operations(parent_data=parent_data)
        )
        return entries

    def display_code(self, *, parent_data: xr.DataArray | None = None) -> str | None:
        """Return streamlined replay code for UI and clipboard actions.

        The display path preserves exact live-source behavior while omitting user-facing
        no-op and normalization steps from copied provenance code.
        """
        prefix: str | None = None
        if self.kind == "script" and self.script_inputs:
            return self._script_graph_code(display=True)
        if self.kind in {"script", "file"}:
            prefix = self.seed_code

        step_codes = self._code_lines_from_entries(
            self.display_entries(parent_data=parent_data)[1:]
        )
        if step_codes is None:
            return None

        if prefix is None and self.kind != "script":
            if not step_codes:
                return None
            prefix = _DEFAULT_REPLAY_SEED_CODE
        if prefix is None and not step_codes:
            return None
        if not step_codes and prefix == _DEFAULT_REPLAY_SEED_CODE:
            return None
        inline_targets = (
            {"derived"} if self.kind == "script" and self.script_inputs else None
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
    return ToolProvenanceSpec.model_validate(value)


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
            if isinstance(operation, (QSelOperation, IselOperation, SelOperation)):
                if operation.decoded_kwargs:
                    break
                continue
            if isinstance(operation, SortCoordOrderOperation):
                continue
            if isinstance(operation, SqueezeOperation):
                saw_squeeze = True
                continue
            break
        else:
            if saw_squeeze:
                return parent_spec

    entries = source.display_entries(parent_data=parent_data)
    local_spec = ToolProvenanceSpec(
        kind="script",
        start_label=entries[0].label,
        seed_code=_DEFAULT_REPLAY_SEED_CODE,
        active_name="derived",
        operations=tuple(
            ScriptCodeOperation(
                label=entry.label,
                code=entry.code,
                copyable=entry.copyable,
            )
            for entry in entries[1:]
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
            operations=tuple(
                ScriptCodeOperation(
                    label=entry.label,
                    code=entry.code,
                    copyable=entry.copyable,
                )
                for entry in spec.derivation_entries()[1:]
            ),
        )
    return spec.to_replay_spec()


def compose_full_provenance(
    parent: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
    local: ToolProvenanceSpec | Mapping[str, typing.Any] | None,
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

    local_operations = list(local_spec.operations)
    if local_spec.seed_code:
        seed_code: str | None = local_spec.seed_code
        if seed_code == _DEFAULT_REPLAY_SEED_CODE:
            parent_input = replay_input_name(parent_spec)
            if parent_input == "derived":
                seed_code = None
            elif parent_input is not None:
                seed_code = f"derived = {parent_input}"
        if seed_code is not None:
            local_operations.insert(
                0,
                ScriptCodeOperation(
                    label=typing.cast("str", local_spec.start_label),
                    code=seed_code,
                ),
            )
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
                ScriptCodeOperation(
                    label="Use current parent output as the active data",
                    code=f"derived = {parent_input}",
                ),
            )

    return ToolProvenanceSpec(
        kind="script",
        start_label=typing.cast("str", parent_spec.start_label),
        seed_code=parent_spec.seed_code,
        active_name=local_spec.active_name or parent_spec.active_name,
        file_load_source=parent_spec.file_load_source or local_spec.file_load_source,
        script_inputs=(*parent_spec.script_inputs, *local_spec.script_inputs),
        operations=(*parent_spec.operations, *local_operations),
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


class ImageToolSelectionSourceBinding(pydantic.BaseModel):
    """ImageTool selection state stored for manager child refreshes.

    Stores the parent dimension indices selected in an ImageTool plot. When a child
    refreshes after parent coordinates change, :meth:`materialize` rebuilds ``qsel`` or
    ``isel`` operations from the current parent data so the child follows the same
    cursor or bin position instead of old coordinate labels.

    Parameters
    ----------
    selection_mode
        Use ``"qsel"`` when selected hidden dimensions should be represented by current
        coordinate values, or ``"isel"`` when they must stay index-based.
    selection_indexers
        Parent dimension names mapped to integer indices or index slices from the
        cursor or bin selection.
    selection_binned_dims
        Dimensions in ``selection_indexers`` whose slices should become ``qsel`` center
        and width arguments.
    crop_sel_indexers
        Index slice selections from visible axes that should be represented by current
        coordinate values during refresh.
    crop_isel_indexers
        Index slice selections from visible non-uniform axes.
    transpose_dims
        Dimension order to apply after selection, if the opened tool expects a
        transposed input.
    squeeze
        Whether to squeeze singleton dimensions after selection.
    """

    schema_version: typing.Literal[1] = 1
    kind: typing.Literal["imagetool_selection"] = "imagetool_selection"
    selection_mode: typing.Literal["qsel", "isel"] = "qsel"
    selection_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    selection_binned_dims: ProvenanceHashableTuple = ()
    crop_sel_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    crop_isel_indexers: ProvenanceMapping = pydantic.Field(default_factory=dict)
    transpose_dims: NullableProvenanceHashableTuple = None
    squeeze: bool = False

    model_config = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def materialize(self, parent_data: xr.DataArray) -> ToolProvenanceSpec:
        """Build a source spec for the current parent data.

        Parameters
        ----------
        parent_data
            Current data from the parent ImageTool.

        Returns
        -------
        ToolProvenanceSpec
            Source spec whose ``qsel``, ``isel``, and ``sel`` operations match this
            binding on ``parent_data``.
        """
        operations: list[ToolProvenanceOperation] = []
        selection_data = ToolProvenanceSpec._starting_data_for_kind(
            "selection", parent_data
        )
        selection_indexers = dict(self.selection_indexers)
        if selection_indexers:
            if self.selection_mode == "qsel":
                operation = QSelOperation(
                    kwargs=erlab.interactive.imagetool.slicer.qsel_args_from_indexers(
                        selection_data,
                        selection_indexers,
                        self.selection_binned_dims,
                    )
                )
            else:
                operation = IselOperation(kwargs=selection_indexers)
            operations.append(operation)

        crop_sel_indexers = dict(self.crop_sel_indexers)
        if crop_sel_indexers:
            crop_sel_kwargs: dict[Hashable, typing.Any] = {}
            for dim, selector in crop_sel_indexers.items():
                if dim not in selection_data.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data")
                coord = selection_data[dim][selector].values
                if isinstance(selector, slice):
                    if coord.size == 0:
                        raise ValueError(f"Selection for dimension `{dim}` is empty")
                    crop_sel_kwargs[dim] = slice(
                        erlab.utils.misc._convert_to_native(coord[0]),
                        erlab.utils.misc._convert_to_native(coord[-1]),
                    )
                else:
                    crop_sel_kwargs[dim] = erlab.utils.misc._convert_to_native(coord)
            operations.append(SelOperation(kwargs=crop_sel_kwargs))

        crop_isel_indexers = dict(self.crop_isel_indexers)
        if crop_isel_indexers:
            operations.append(IselOperation(kwargs=crop_isel_indexers))

        operations.append(SortCoordOrderOperation())
        if self.transpose_dims is not None:
            operations.append(TransposeOperation(dims=self.transpose_dims))
        if self.squeeze:
            operations.append(SqueezeOperation())
        return ToolProvenanceSpec(kind="selection").append_operations(*operations)


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
    script_inputs: Sequence[ScriptInput] = (),
) -> ToolProvenanceSpec:
    """Build script provenance from code, structured steps, and named inputs."""
    return ToolProvenanceSpec(
        kind="script",
        start_label=start_label,
        seed_code=seed_code,
        active_name=active_name,
        file_load_source=file_load_source,
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


def _parse_replay_dataset(ds: xr.Dataset) -> tuple[xr.DataArray, ...]:
    return tuple(
        darr for darr in ds.data_vars.values() if _supported_replay_shape(darr)
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
    parsed = _parse_replay_input(loaded)
    if call.selected_index >= len(parsed):
        raise IndexError(
            f"Selected file replay index {call.selected_index} is out of range for "
            f"{len(parsed)} parsed arrays"
        )
    data = parsed[call.selected_index]
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

    for node in ast.walk(module):
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


def replay_script_provenance(
    spec: ToolProvenanceSpec | Mapping[str, typing.Any],
    inputs: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """Execute script provenance from already resolved input arrays.

    The caller is responsible for trust and input resolution. This function only
    validates the provenance shape, compiles it through the replay graph, and returns
    the replayed :class:`xarray.DataArray`.
    """
    try:
        return _replay_graph.replay_script_provenance(spec, inputs)
    except _replay_graph.ReplayGraphError as exc:
        if "non-replayable" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(str(exc)) from exc


class ScriptCodeOperation(ToolProvenanceOperation):
    op: typing.Literal["script_code"] = "script_code"
    label: str
    code: str | None
    copyable: bool = True

    live_applicable: typing.ClassVar[bool] = False

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        raise TypeError(
            "script_code operations do not support live updates from ImageTool data"
        )

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.label, self.code, self.copyable)


class QSelOperation(ToolProvenanceOperation):
    op: typing.Literal["qsel"] = "qsel"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            accessor_path=("qsel",),
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"qsel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("qsel", kwargs),
            True,
        )


class IselOperation(ToolProvenanceOperation):
    op: typing.Literal["isel"] = "isel"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="isel",
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
            ignored_defaults={"drop": False, "missing_dims": "raise"},
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.isel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"isel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("isel", kwargs),
            True,
        )


class SelOperation(ToolProvenanceOperation):
    op: typing.Literal["sel"] = "sel"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="sel",
            kwargs_field="kwargs",
            mapping_kwarg="indexers",
            ignored_defaults={"method": None, "tolerance": None, "drop": False},
        ),
    )
    kwargs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    @property
    def decoded_kwargs(self) -> dict[Hashable, typing.Any]:
        return dict(self.kwargs)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.sel(self.decoded_kwargs)

    def derivation_entry(self) -> DerivationEntry:
        kwargs = self.decoded_kwargs
        return DerivationEntry(
            f"sel({_format_derivation_value(self.kwargs)})",
            _format_selection_step("sel", kwargs),
            True,
        )


class SortCoordOrderOperation(ToolProvenanceOperation):
    op: typing.Literal["sort_coord_order"] = "sort_coord_order"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(data, parent_data.coords.keys())

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            _SORT_COORD_ORDER_DERIVATION_LABEL,
            _SORT_COORD_ORDER_DERIVATION_CODE,
            True,
        )


class SelectCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["select_coord"] = "select_coord"
    coord_name: ProvenanceHashable

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.coords[self.coord_name].copy(deep=False)

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        label_kwargs = {"coord_name": self.coord_name}
        return DerivationEntry(
            f"Select Coordinate({_format_derivation_value(label_kwargs)})",
            f"derived = derived.coords[{coord_name_code}]",
            True,
        )


class TransposeOperation(ToolProvenanceOperation):
    op: typing.Literal["transpose"] = "transpose"
    dims: NullableProvenanceHashableTuple = None

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "transpose"
            or call.accessor_path
        ):
            return None
        kwargs = dict(call.kwargs)
        for key, default in {"transpose_coords": True, "missing_dims": "raise"}.items():
            if key not in kwargs:
                continue
            if not _console_values_equal(kwargs[key], default):
                return None
            kwargs.pop(key)
        if kwargs:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dims=tuple(call.args) or None)
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.dims:
            return data.transpose(*self.dims)
        return data.transpose(*reversed(data.dims))

    def derivation_entry(self) -> DerivationEntry:
        if self.dims:
            dims_tuple = tuple(self.dims)
            return DerivationEntry(
                f"transpose({_format_derivation_value(dims_tuple)})",
                "derived = derived.transpose(*"
                f"{erlab.interactive.utils._parse_single_arg(dims_tuple)})",
                True,
            )
        return DerivationEntry(
            "transpose()",
            "derived = derived.transpose(*reversed(derived.dims))",
            True,
        )


class SqueezeOperation(ToolProvenanceOperation):
    op: typing.Literal["squeeze"] = "squeeze"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "squeeze"
            or call.accessor_path
            or call.args
        ):
            return None
        for key, default in {"dim": None, "axis": None, "drop": False}.items():
            if key not in call.kwargs:
                continue
            if not _console_values_equal(call.kwargs[key], default):
                return None
        if set(call.kwargs).difference({"dim", "axis", "drop"}):
            return None
        return cls()

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.squeeze()

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry("squeeze()", "derived = derived.squeeze()", True)


class RenameOperation(ToolProvenanceOperation):
    op: typing.Literal["rename"] = "rename"
    name: str

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "rename"
            or call.accessor_path
            or call.kwargs
            or len(call.args) != 1
            or isinstance(call.args[0], Mapping)
        ):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(name=call.args[0])
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.rename(self.name)

    def derivation_entry(self) -> DerivationEntry:
        name_code = erlab.interactive.utils._parse_single_arg(self.name)
        return DerivationEntry(
            f"rename({_format_derivation_value(self.name)})",
            f"derived = derived.rename({name_code})",
            True,
        )


class RestoreNonuniformDimsOperation(ToolProvenanceOperation):
    op: typing.Literal["restore_nonuniform_dims"] = "restore_nonuniform_dims"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.interactive.imagetool.slicer.restore_nonuniform_dims(data)

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            "Restore nonuniform dimensions",
            "derived = erlab.interactive.imagetool.slicer."
            "restore_nonuniform_dims(derived)",
            True,
        )


class RotateOperation(ToolProvenanceOperation):
    op: typing.Literal["rotate"] = "rotate"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.rotate",
            fields=("angle", "axes", "center"),
            defaults={
                "axes": (0, 1),
                "center": (0.0, 0.0),
                "reshape": True,
                "order": 1,
            },
            ignored_defaults={"mode": "constant", "cval": np.nan, "prefilter": True},
        ),
    )
    angle: float
    axes: ProvenanceHashablePair
    center: tuple[float, float]
    reshape: bool = True
    order: int = 1

    @pydantic.field_validator("center", mode="before")
    @classmethod
    def _validate_center(cls, value: typing.Any) -> tuple[float, float]:
        return typing.cast(
            "tuple[float, float]",
            _ensure_float_tuple(
                typing.cast("Sequence[typing.Any]", value), expected_len=2
            ),
        )

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "angle": self.angle,
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.rotate(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.transform.rotate,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Rotate({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


def _format_qsel_dims_arg(dims: ProvenanceHashableTuple) -> str:
    return (
        erlab.interactive.utils._parse_single_arg(dims[0])
        if len(dims) == 1 and isinstance(dims[0], str)
        else erlab.interactive.utils._parse_single_arg(dims)
    )


class AverageOperation(ToolProvenanceOperation):
    op: typing.Literal["average"] = "average"
    dims: ProvenanceHashableTuple

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method is not None
            or call.accessor_path != ("qsel", "average")
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        if call.args:
            if "dim" in kwargs:
                return None
            dim = call.args[0]
        else:
            dim = kwargs.pop("dim", None)
        if kwargs or dim is None:
            return None
        dims = (dim,) if isinstance(dim, str) or not isinstance(dim, Sequence) else dim
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dims=typing.cast("tuple[Hashable, ...]", dims))
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.qsel.mean(self.dims)

    def derivation_entry(self) -> DerivationEntry:
        arg = _format_qsel_dims_arg(self.dims)
        label_kwargs = {"dims": self.dims}
        return DerivationEntry(
            f"Average({_format_derivation_value(label_kwargs)})",
            f"derived = derived.qsel.mean({arg})",
            True,
        )


class QSelAggregationOperation(ToolProvenanceOperation):
    op: typing.Literal["qsel_aggregate"] = "qsel_aggregate"
    dims: ProvenanceHashableTuple
    func: typing.Literal["mean", "min", "max", "sum"] = "mean"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        func = call.accessor_path[1] if len(call.accessor_path) == 2 else None
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method is not None
            or len(call.accessor_path) != 2
            or call.accessor_path[0] != "qsel"
            or func not in {"mean", "min", "max", "sum"}
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        if call.args:
            if "dim" in kwargs:
                return None
            dim = call.args[0]
        else:
            dim = kwargs.pop("dim", None)
        if kwargs or dim is None:
            return None
        dims = (dim,) if isinstance(dim, str) or not isinstance(dim, Sequence) else dim
        func = typing.cast("typing.Literal['mean', 'min', 'max', 'sum']", func)
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(
                dims=typing.cast("tuple[Hashable, ...]", dims),
                func=func,
            )
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return typing.cast("xr.DataArray", getattr(data.qsel, self.func)(self.dims))

    def derivation_entry(self) -> DerivationEntry:
        arg = _format_qsel_dims_arg(self.dims)
        label_kwargs = {"dims": self.dims, "func": self.func}
        return DerivationEntry(
            f"Aggregate({_format_derivation_value(label_kwargs)})",
            f"derived = derived.qsel.{self.func}({arg})",
            True,
        )


class InterpolationOperation(ToolProvenanceOperation):
    op: typing.Literal["interpolate"] = "interpolate"
    dim: ProvenanceHashable
    values: typing.Any
    method: typing.Literal["linear", "nearest"] = "linear"

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "interp"
            or call.accessor_path
        ):
            return None
        kwargs = dict(call.kwargs)
        method = kwargs.pop("method", "linear")
        if method not in {"linear", "nearest"}:
            return None
        for key, default in {"assume_sorted": False, "kwargs": None}.items():
            if key not in kwargs:
                continue
            if not _console_values_equal(kwargs[key], default):
                return None
            kwargs.pop(key)
        coords = _console_mapping_values(call.args, kwargs, mapping_kwargs=("coords",))
        if coords is None:
            return None
        if len(coords) != 1:
            return None
        ((dim, values),) = coords.items()
        if not isinstance(dim, Hashable):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(dim=dim, values=values, method=method)
        return None

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        encoded = cls._validate_encoded_field(
            value,
            error="interpolation values must be array-like",
            predicate=lambda encoded_value: (
                isinstance(encoded_value, list)
                or (
                    isinstance(encoded_value, Mapping)
                    and _TUPLE_MARKER in encoded_value
                )
            ),
        )
        values = np.asarray(cls._decode_stored_field(encoded))
        if values.ndim != 1:
            raise ValueError("interpolation values must be one-dimensional.")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise TypeError("interpolation values must be real numeric values.")
        if not np.all(np.isfinite(values.astype(np.float64, copy=False))):
            raise ValueError("interpolation values must be finite.")
        return encoded

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.interp({self.dim: self.decoded_values}, method=self.method)

    def code(self, data_name: str, *, assign: str | None = None) -> str:
        dim_code = erlab.interactive.utils._parse_single_arg(self.dim)
        values_code = erlab.interactive.utils.format_1d_numeric_array_code(
            self.decoded_values
        )
        method_code = erlab.interactive.utils._parse_single_arg(self.method)
        code = (
            f"{data_name}.interp({{{dim_code}: {values_code}}}, method={method_code})"
        )
        if assign is not None:
            return f"{assign} = {code}"
        return code

    def derivation_entry(self) -> DerivationEntry:
        label_kwargs = {
            "dim": self.dim,
            "values": self.values,
            "method": self.method,
        }
        return DerivationEntry(
            f"Interpolate({_format_derivation_value(label_kwargs)})",
            self.code("derived", assign="derived"),
            True,
        )


class LeadingEdgeOperation(ToolProvenanceOperation):
    op: typing.Literal["leading_edge"] = "leading_edge"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.interpolate.leading_edge",
            fields=("fraction", "dim", "direction"),
            defaults={
                "fraction": 0.5,
                "dim": "eV",
                "direction": "positive",
            },
        ),
    )
    fraction: float = pydantic.Field(default=0.5, gt=0.0, le=1.0)
    dim: ProvenanceHashable = "eV"
    direction: typing.Literal["positive", "negative"] = "positive"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fraction": self.fraction,
            "dim": self.dim,
            "direction": self.direction,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.leading_edge(data, **self.kwargs)

    def code(self, data_name: str, *, assign: str | None = None) -> str:
        return erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.leading_edge,
            [f"|{data_name}|"],
            self.kwargs,
            module="era.interpolate",
            assign=assign,
        )

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            f"Leading Edge({_format_derivation_value(self.kwargs)})",
            self.code("derived", assign="derived"),
            True,
        )


class DivideByCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["divide_by_coord"] = "divide_by_coord"
    coord_name: ProvenanceHashable

    @staticmethod
    def _raise_if_zero(coord: xr.DataArray) -> None:
        if np.any(np.asarray(coord.values) == 0):
            raise ValueError("Coordinate contains zero values and cannot be a divisor.")

    def divisor_code(self, data_name: str) -> str:
        if (
            isinstance(self.coord_name, str)
            and self.coord_name.isidentifier()
            and not keyword.iskeyword(self.coord_name)
            and not self.coord_name.startswith("_")
            and not hasattr(xr.DataArray, self.coord_name)
        ):
            return f"{data_name}.{self.coord_name}"
        coord_name = erlab.interactive.utils._parse_single_arg(self.coord_name)
        return f"{data_name}.coords[{coord_name}]"

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        coord = data.coords[self.coord_name]
        self._raise_if_zero(coord)
        return data / coord

    def derivation_entry(self) -> DerivationEntry:
        label_kwargs = {"coord_name": self.coord_name}
        return DerivationEntry(
            f"Divide by Coordinate({_format_derivation_value(label_kwargs)})",
            f"derived = derived / {self.divisor_code('derived')}",
            True,
        )


class CoarsenOperation(ToolProvenanceOperation):
    op: typing.Literal["coarsen"] = "coarsen"
    dim: ProvenanceIntMapping = pydantic.Field(default_factory=dict)
    boundary: str
    side: str
    coord_func: str
    reducer: str

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "coarsen"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        kwargs = dict(call.kwargs)
        reducer = kwargs.pop("_reducer", None)
        if not isinstance(reducer, str):
            return None
        dim: dict[typing.Any, typing.Any] = {}
        if call.args:
            if not isinstance(call.args[0], Mapping):
                return None
            dim.update(call.args[0])
        dim.update(kwargs.pop("dim", {}) or {})
        dim.update(
            {
                key: kwargs.pop(key)
                for key in list(kwargs)
                if key not in {"boundary", "side", "coord_func"}
            }
        )
        values = {
            "dim": dim,
            "boundary": kwargs.pop("boundary", "exact"),
            "side": kwargs.pop("side", "left"),
            "coord_func": kwargs.pop("coord_func", "mean"),
            "reducer": reducer,
        }
        if kwargs:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(**values)
        return None

    @property
    def coarsen_kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "boundary": self.boundary,
            "side": self.side,
            "coord_func": self.coord_func,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        coarsened = data.coarsen(**self.coarsen_kwargs)
        return typing.cast("xr.DataArray", getattr(coarsened, self.reducer)())

    def derivation_entry(self) -> DerivationEntry:
        coarsen_kwargs: Mapping[typing.Any, typing.Any]
        if _is_identifier_string_mapping(self.dim):
            coarsen_kwargs = {
                **self.dim,
                "boundary": self.boundary,
                "side": self.side,
                "coord_func": self.coord_func,
            }
        else:
            coarsen_kwargs = self.coarsen_kwargs
        coarsen_kwargs = erlab.interactive.utils._remove_default_kwargs(
            xr.DataArray.coarsen, dict(coarsen_kwargs)
        )
        formatted_kwargs = erlab.interactive.utils.format_kwargs(coarsen_kwargs)
        code = f"derived = derived.coarsen({formatted_kwargs}).{self.reducer}()"
        label_kwargs = {"coarsen_kwargs": self.coarsen_kwargs, "reducer": self.reducer}
        return DerivationEntry(
            f"Coarsen({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class ThinOperation(ToolProvenanceOperation):
    op: typing.Literal["thin"] = "thin"
    mode: typing.Literal["global", "per_dim"]
    factor: int | None = None
    factors: ProvenanceIntMapping = pydantic.Field(default_factory=dict)

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "thin"
            or call.accessor_path
        ):
            return None
        if (
            len(call.args) == 1
            and not call.kwargs
            and not isinstance(call.args[0], Mapping)
        ):
            with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
                return cls(mode="global", factor=call.args[0])
            return None
        if len(call.args) > 1:
            return None
        factors: typing.Any = dict(call.kwargs)
        if call.args:
            if call.args[0] is None:
                pass
            elif isinstance(call.args[0], Mapping):
                factors = {**call.args[0], **factors}
            else:
                return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(mode="per_dim", factors=factors)
        return None

    @pydantic.model_validator(mode="after")
    def _check_mode(self) -> typing.Self:
        if self.mode == "global" and self.factor is None:
            raise ValueError("thin global mode requires factor")
        if self.mode == "per_dim" and not self.factors:
            raise ValueError("thin per_dim mode requires factors")
        return self

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        if self.mode == "global":
            return {"mode": self.mode, "factor": int(typing.cast("int", self.factor))}
        return {"mode": self.mode, "factors": self.factors}

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        if self.mode == "global":
            return data.thin(int(typing.cast("int", self.factor)))
        return data.thin(self.factors)

    def derivation_entry(self) -> DerivationEntry:
        if self.mode == "global":
            code = f"derived = derived.thin({int(typing.cast('int', self.factor))})"
        else:
            code = (
                "derived = derived.thin("
                f"{erlab.interactive.utils.format_call_kwargs(self.factors)})"
            )
        return DerivationEntry(
            f"Thin({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class SymmetrizeOperation(ToolProvenanceOperation):
    op: typing.Literal["symmetrize"] = "symmetrize"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.symmetrize",
            fields=("dim",),
            defaults={
                "center": 0.0,
                "subtract": False,
                "mode": "full",
                "part": "both",
            },
            ignored_defaults={"interp_kw": None},
        ),
    )
    dim: ProvenanceHashable
    center: float
    subtract: bool = False
    mode: typing.Literal["full", "valid"] = "full"
    part: typing.Literal["both", "below", "above"] = "both"

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self.dim,
            "center": self.center,
            "subtract": self.subtract,
            "mode": self.mode,
            "part": self.part,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Symmetrize({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class SymmetrizeNfoldOperation(ToolProvenanceOperation):
    op: typing.Literal["symmetrize_nfold"] = "symmetrize_nfold"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.transform.symmetrize_nfold",
            fields=("fold", "axes", "center"),
            defaults={
                "axes": (0, 1),
                "center": (0.0, 0.0),
                "reshape": True,
                "order": 1,
            },
            ignored_defaults={"mode": "constant", "cval": np.nan, "prefilter": True},
        ),
    )
    fold: int
    axes: ProvenanceHashablePair
    center: typing.Any = (0.0, 0.0)
    reshape: bool = True
    order: int = 1

    @pydantic.field_validator("center", mode="before")
    @classmethod
    def _validate_center(cls, value: typing.Any) -> typing.Any:
        decoded = decode_provenance_value(value)
        if isinstance(decoded, Mapping):
            return _coerce_float_mapping_field(decoded)
        return _ensure_float_tuple(
            typing.cast("Sequence[typing.Any]", decoded), expected_len=2
        )

    @pydantic.field_serializer("center", when_used="json")
    def _serialize_center(self, value: typing.Any) -> typing.Any:
        return encode_provenance_value(value)

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "fold": self.fold,
            "axes": self.axes,
            "center": self.center,
            "reshape": self.reshape,
            "order": self.order,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.transform.symmetrize_nfold(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.transform.symmetrize_nfold,
            ["|derived|"],
            self.kwargs,
            module="era.transform",
            assign="derived",
        )
        return DerivationEntry(
            f"Rotational Symmetrize({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class CorrectWithEdgeOperation(ToolProvenanceOperation):
    op: typing.Literal["correct_with_edge"] = "correct_with_edge"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.gold.correct_with_edge",
            fields=("edge_fit",),
            field_aliases={"modelresult": "edge_fit"},
            defaults={"shift_coords": True},
        ),
    )
    edge_fit: typing.Any
    shift_coords: bool = True

    @pydantic.field_validator("edge_fit", mode="before")
    @classmethod
    def _validate_edge_fit(cls, value: typing.Any) -> typing.Any:
        if isinstance(value, Mapping) and (
            _DATASET_MARKER in value or _FIT_DATASET_MARKER in value
        ):
            return value
        return cls._validate_encoded_field(
            value,
            error="correct_with_edge edge_fit must be an xarray.Dataset",
            predicate=lambda encoded: (
                isinstance(encoded, Mapping)
                and (_DATASET_MARKER in encoded or _FIT_DATASET_MARKER in encoded)
            ),
        )

    @property
    def decoded_edge_fit(self) -> xr.Dataset:
        return typing.cast("xr.Dataset", self._decode_stored_field(self.edge_fit))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.gold.correct_with_edge(
            data,
            self.decoded_edge_fit,
            shift_coords=self.shift_coords,
        )

    def derivation_entry(self) -> DerivationEntry:
        label_kwargs = {
            "edge_fit": self.edge_fit,
            "shift_coords": self.shift_coords,
        }
        edge_fit_code = _provenance_value_code(self.edge_fit)
        edge_fit_expr = (
            "erlab.interactive.imagetool.provenance.decode_provenance_value("
            f"{edge_fit_code})"
        )
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.gold.correct_with_edge,
            ["|derived|", f"|{edge_fit_expr}|"],
            {"shift_coords": self.shift_coords},
            module="era.gold",
            name="correct_with_edge",
            assign="derived",
        )
        return DerivationEntry(
            f"Edge Correction({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class SwapDimsOperation(ToolProvenanceOperation):
    op: typing.Literal["swap_dims"] = "swap_dims"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            dataarray_method="swap_dims",
            kwargs_field="mapping",
            mapping_kwarg="dims_dict",
        ),
    )
    mapping: ProvenanceHashableMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.swap_dims(self.mapping)

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            f"Swap Dimensions({_format_derivation_value(self.mapping)})",
            "derived = derived.swap_dims("
            f"{erlab.interactive.utils.format_call_kwargs(self.mapping)})",
            True,
        )


class RenameDimsCoordsOperation(ToolProvenanceOperation):
    op: typing.Literal["rename_dims_coords"] = "rename_dims_coords"
    mapping: ProvenanceHashableMapping = pydantic.Field(default_factory=dict)

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        if (
            call.has_extra_tracked_inputs
            or call.dataarray_method != "rename"
            or call.accessor_path
            or len(call.args) > 1
        ):
            return None
        mapping = _console_mapping_values(
            call.args, call.kwargs, mapping_kwargs=("new_name_or_name_dict",)
        )
        if mapping is None:
            return None
        if not mapping:
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(mapping=mapping)
        return None

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.rename(self.mapping)

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(
            f"Rename({_format_derivation_value(self.mapping)})",
            "derived = derived.rename("
            f"{erlab.interactive.utils.format_call_kwargs(self.mapping)})",
            True,
        )


class AffineCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["affine_coord"] = "affine_coord"
    coord_name: str
    scale: float
    offset: float

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        coord = data.coords[self.coord_name]
        return erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {
                    self.coord_name: coord.copy(
                        data=self.scale * coord.values + self.offset
                    )
                }
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = repr(self.coord_name)
        scale_code = erlab.interactive.utils._parse_single_arg(float(self.scale))
        offset_code = erlab.interactive.utils._parse_single_arg(float(self.offset))
        code = (
            f"derived = derived.assign_coords({{{coord_name_code}: "
            f"derived[{coord_name_code}].copy(data={scale_code} * "
            f"derived[{coord_name_code}].values + {offset_code})}})"
        )
        label_kwargs = {
            "coord_name": self.coord_name,
            "scale": self.scale,
            "offset": self.offset,
        }
        return DerivationEntry(
            f"Scale/Offset Coordinate({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class AssignCoordsOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_coords"] = "assign_coords"
    coord_name: str
    values: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None or call.receiver_data is None:
            return None
        coord_name, values = coord
        if (
            not isinstance(coord_name, str)
            or coord_name not in call.receiver_data.coords
        ):
            return None
        if (
            isinstance(values, Sequence)
            and not isinstance(values, str)
            and len(values) == 2
            and isinstance(values[0], Hashable)
        ):
            return None
        if _scalar_coord_value(values):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, values=values)
        return None

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="assign_coords values must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(
            data.assign_coords(
                {self.coord_name: data[self.coord_name].copy(data=self.decoded_values)}
            ),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = repr(self.coord_name)
        values = self.decoded_values
        values_code = _provenance_numeric_array_code(values)
        code = (
            f"derived = derived.assign_coords({{{coord_name_code}: "
            f"derived[{coord_name_code}].copy(data={values_code})}})"
        )
        label_kwargs = {
            "coord_name": self.coord_name,
            "values": self.values,
        }
        return DerivationEntry(
            f"Assign Coordinates({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class AssignScalarCoordOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_scalar_coord"] = "assign_scalar_coord"
    coord_name: ProvenanceHashable
    value: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None:
            return None
        coord_name, value = coord
        if not _scalar_coord_value(value):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, value=value)
        return None

    @pydantic.field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: typing.Any) -> typing.Any:
        encoded = encode_provenance_value(value)
        decoded = decode_provenance_value(encoded)
        if isinstance(decoded, (Mapping, Sequence)) and not isinstance(decoded, str):
            raise TypeError("scalar coordinate value must be scalar")
        return encoded

    @property
    def decoded_value(self) -> typing.Any:
        return self._decode_stored_field(self.value)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.utils.array.sort_coord_order(
            data.assign_coords({self.coord_name: self.decoded_value}),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        value_code = _provenance_value_code(self.decoded_value)
        code = f"derived = derived.assign_coords({{{coord_name_code}: {value_code}}})"
        label_kwargs = {
            "coord_name": self.coord_name,
            "value": self.value,
        }
        return DerivationEntry(
            f"Assign Scalar Coordinate({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class AssignCoord1DOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_coord_1d"] = "assign_coord_1d"
    coord_name: ProvenanceHashable
    dim: ProvenanceHashable
    values: typing.Any

    @classmethod
    def from_console_call(cls, call: ConsoleCall) -> ToolProvenanceOperation | None:
        coord = _single_assign_coord(call)
        if coord is None:
            return None
        coord_name, value = coord
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
            return None
        dim, values = value
        if not isinstance(dim, Hashable) or (
            isinstance(dim, Sequence) and not isinstance(dim, str)
        ):
            return None
        try:
            if np.asarray(values).ndim != 1:
                return None
        except (TypeError, ValueError):
            return None
        with contextlib.suppress(TypeError, ValueError, pydantic.ValidationError):
            return cls(coord_name=coord_name, dim=dim, values=values)
        return None

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="1D coordinate values must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def decoded_values(self) -> np.ndarray:
        return np.asarray(self._decode_stored_field(self.values))

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        values = self.decoded_values
        if values.ndim != 1:
            raise ValueError("1D coordinate values must be one-dimensional.")
        if self.dim not in data.sizes:
            raise ValueError(f"Dimension {self.dim!r} is not present in the data.")
        if values.size != data.sizes[self.dim]:
            raise ValueError(
                f"Coordinate {self.coord_name!r} has length {values.size}, expected "
                f"{data.sizes[self.dim]} for dimension {self.dim!r}."
            )
        return erlab.utils.array.sort_coord_order(
            data.assign_coords({self.coord_name: (self.dim, values)}),
            keys=data.coords.keys(),
            dims_first=False,
        )

    def derivation_entry(self) -> DerivationEntry:
        coord_name_code = erlab.interactive.utils._parse_single_arg(self.coord_name)
        dim_code = erlab.interactive.utils._parse_single_arg(self.dim)
        values_code = _provenance_numeric_array_code(self.decoded_values)
        code = (
            f"derived = derived.assign_coords({{{coord_name_code}: "
            f"({dim_code}, {values_code})}})"
        )
        label_kwargs = {
            "coord_name": self.coord_name,
            "dim": self.dim,
            "values": self.values,
        }
        return DerivationEntry(
            f"Assign 1D Coordinate({_format_derivation_value(label_kwargs)})",
            code,
            True,
        )


class AssignAttrsOperation(ToolProvenanceOperation):
    op: typing.Literal["assign_attrs"] = "assign_attrs"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(dataarray_method="assign_attrs", kwargs_field="attrs"),
    )
    attrs: ProvenanceMapping = pydantic.Field(default_factory=dict)

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return data.assign_attrs(self.attrs)

    def derivation_entry(self) -> DerivationEntry:
        attrs_code = _provenance_value_code(self.attrs)
        return DerivationEntry(
            f"Assign Attributes({_format_derivation_value(self.attrs)})",
            f"derived = derived.assign_attrs({attrs_code})",
            True,
        )


class SliceAlongPathOperation(ToolProvenanceOperation):
    op: typing.Literal["slice_along_path"] = "slice_along_path"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.interpolate.slice_along_path",
            fields=("vertices", "step_size", "dim_name"),
            defaults={"dim_name": "path"},
            ignored_defaults={"interp_kwargs": None},
        ),
    )
    vertices: ProvenanceFloatSequenceMapping = pydantic.Field(default_factory=dict)
    step_size: float
    dim_name: str

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": self.vertices,
            "step_size": self.step_size,
            "dim_name": self.dim_name,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.interpolate.slice_along_path(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.interpolate.slice_along_path,
            ["|derived|"],
            self.kwargs,
            module="era.interpolate",
            assign="derived",
        )
        return DerivationEntry(
            f"Slice Along ROI Path({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )


class MaskWithPolygonOperation(ToolProvenanceOperation):
    op: typing.Literal["mask_with_polygon"] = "mask_with_polygon"
    console_patterns: typing.ClassVar[tuple[ConsoleOperationPattern, ...]] = (
        ConsoleOperationPattern(
            target="erlab.analysis.mask.mask_with_polygon",
            fields=("vertices", "dims"),
            defaults={"invert": False, "drop": False},
        ),
    )
    vertices: typing.Any
    dims: ProvenanceHashableTuple
    invert: bool = False
    drop: bool = False

    @pydantic.field_validator("vertices", mode="before")
    @classmethod
    def _validate_vertices(cls, value: typing.Any) -> typing.Any:
        return cls._validate_encoded_field(
            value,
            error="mask_with_polygon vertices must be array-like",
            predicate=lambda encoded: (
                isinstance(encoded, list)
                or (isinstance(encoded, Mapping) and _TUPLE_MARKER in encoded)
            ),
        )

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return {
            "vertices": np.asarray(self._decode_stored_field(self.vertices)),
            "dims": self.dims,
            "invert": self.invert,
            "drop": self.drop,
        }

    def apply(self, data: xr.DataArray, *, parent_data: xr.DataArray) -> xr.DataArray:
        return erlab.analysis.mask.mask_with_polygon(data, **self.kwargs)

    def derivation_entry(self) -> DerivationEntry:
        code = erlab.interactive.utils.generate_code(
            erlab.analysis.mask.mask_with_polygon,
            ["|derived|"],
            self.kwargs,
            module="era.mask",
            assign="derived",
        )
        return DerivationEntry(
            f"Mask with ROI({_format_derivation_value(self.kwargs)})",
            code,
            True,
        )
