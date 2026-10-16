"""Code generation and restricted-script policy for ImageTool provenance."""

from __future__ import annotations

import ast
import keyword
import math
import symtable
import typing
from collections import Counter
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence

import numpy as np

import erlab.utils._code
import erlab.utils.misc

if typing.TYPE_CHECKING:
    import types

_LEGACY_NONUNIFORM_RESTORE_PATH = (
    "erlab",
    (
        "interactive",
        "imagetool",
        "slicer",
        "restore_nonuniform_dims",
    ),
)

_PRIVATE_NONUNIFORM_RESTORE_EXPRESSION = "erlab.utils.array._restore_nonuniform_dims"

_SLICE_MARKER = "__erlab_slice__"

_DATASET_MARKER = "__erlab_xarray_dataset__"

_FIT_DATASET_MARKER = "__erlab_xarray_lmfit_dataset__"

_DATAARRAY_MARKER = "__erlab_xarray_dataarray__"

_TUPLE_MARKER = "__erlab_tuple__"

_MAPPING_MARKER = "__erlab_mapping__"

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
    "TypeError": TypeError,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
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
        return erlab.utils._code.format_1d_numeric_array_code(values)
    return _provenance_value_code(values)


def _format_selection_expr(
    input_name: str, method: str, kwargs: Mapping[Hashable, typing.Any]
) -> str:
    if not kwargs:
        return f"{input_name}.{method}()"
    args = erlab.utils._code.format_kwargs(dict(kwargs))
    return f"{input_name}.{method}({args})"


def _format_selection_step(method: str, kwargs: Mapping[Hashable, typing.Any]) -> str:
    return f"derived = {_format_selection_expr('derived', method, kwargs)}"


def _nonuniform_dim_mapping_condition(
    array_name: str,
    index_code: str,
    restored_code: str,
) -> str:
    return (
        f"{index_code} in {array_name}.dims and "
        f"{restored_code} in {array_name}.coords and "
        f"{array_name}.coords[{restored_code}].ndim == 1 and "
        f"{array_name}.coords[{restored_code}].dims == ({index_code},) and "
        f"{array_name}.coords[{restored_code}].size == "
        f"{array_name}.sizes[{index_code}]"
    )


def _restore_nonuniform_dims_expression(
    input_name: str,
    dimension_mapping: Mapping[Hashable, Hashable],
) -> str:
    """Emit a self-contained expression restoring recorded index dimensions.

    This is used for concise copied selection expressions whose receiver is a stable
    data variable. Applicable dimensions are restored together so expression size grows
    linearly with the recorded mapping instead of nesting the preceding expression for
    every dimension. Structured provenance operations use statement code when the
    receiver itself may need single evaluation.
    """
    if not dimension_mapping:
        return input_name

    receiver = _expression_receiver_code(input_name)
    mapping_items = ", ".join(
        f"({_provenance_value_code(index_dim)}, {_provenance_value_code(restored_dim)})"
        for index_dim, restored_dim in dimension_mapping.items()
    )
    mapping_code = f"({mapping_items},)"
    condition = _nonuniform_dim_mapping_condition(
        receiver, "index_dimension", "dimension"
    )
    applicable_mapping = (
        "{index_dimension: dimension "
        f"for index_dimension, dimension in {mapping_code} if {condition}}}"
    )
    obsolete_dimensions = (
        "[index_dimension "
        f"for index_dimension, dimension in {mapping_code} if {condition}]"
    )
    return (
        f"{receiver}.swap_dims({applicable_mapping}).drop_vars("
        f'{obsolete_dimensions}, errors="ignore")'
    )


def _known_nonuniform_restore_statement_code(
    input_name: str,
    *,
    output_name: str,
    dimension_mapping: Mapping[Hashable, Hashable],
) -> str:
    """Emit public xarray statements restoring recorded index dimensions."""
    lines = (
        []
        if input_name == output_name and dimension_mapping
        else [f"{output_name} = {input_name}"]
    )
    for index_dim, restored_dim in dimension_mapping.items():
        index_code = _provenance_value_code(index_dim)
        restored_code = _provenance_value_code(restored_dim)
        condition = _nonuniform_dim_mapping_condition(
            output_name, index_code, restored_code
        )
        lines.extend(
            (
                f"if {condition}:",
                f"    {output_name} = {output_name}.swap_dims("
                f"{{{index_code}: {restored_code}}}).drop_vars("
                f'{index_code}, errors="ignore")',
            )
        )
    return "\n".join(lines)


_NONUNIFORM_RESTORE_FUNCTION_NAME = "_restore_image_tool_dimensions"


def _nonuniform_restore_support_code(
    function_name: str = _NONUNIFORM_RESTORE_FUNCTION_NAME,
) -> str:
    """Emit standalone public-API code for detecting ImageTool index dimensions."""
    return f'''def {function_name}(array):
    """Restore nonuniform dimensions replaced for ImageTool rendering."""
    import numpy as np

    dimension_mapping = {{}}
    for index_dimension in array.dims:
        if not str(index_dimension).endswith("_idx"):
            continue
        dimension = str(index_dimension).removesuffix("_idx")
        coordinate = array.coords.get(dimension)
        if coordinate is None:
            continue
        if (
            coordinate.ndim != 1
            or coordinate.dims != (index_dimension,)
            or coordinate.size != array.sizes[index_dimension]
        ):
            continue
        try:
            values = coordinate.values.astype(np.float64)
        except (TypeError, ValueError):
            continue
        if values.size == 1:
            continue
        differences = np.diff(values)
        if differences[0] == 0.0 or not np.allclose(
            differences,
            differences[0],
            rtol=3e-5,
            atol=3e-5,
            equal_nan=True,
        ):
            dimension_mapping[index_dimension] = dimension
    if not dimension_mapping:
        return array
    return array.swap_dims(dimension_mapping).drop_vars(
        tuple(dimension_mapping), errors="ignore"
    )'''


def _dynamic_nonuniform_restore_replay_code(
    input_name: str,
    *,
    output_name: str,
    copy_input: bool = False,
) -> str:
    """Emit a standalone dynamic restore for legacy derivation-code fallbacks."""
    input_expression = f"{input_name}.copy(deep=False)" if copy_input else input_name
    return "\n".join(
        (
            _nonuniform_restore_support_code(),
            f"{output_name} = {_NONUNIFORM_RESTORE_FUNCTION_NAME}({input_expression})",
        )
    )


def _validate_active_name(value: typing.Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("active_name must be a string or None")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError("active_name must be a valid Python identifier")
    return value


def _symbol_table_dependencies(
    table: symtable.SymbolTable, enclosing_bound: frozenset[str] = frozenset()
) -> set[str]:
    """Return external reads, excluding closures supplied by enclosing scopes."""
    dependencies = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_referenced()
        and (
            symbol.is_global()
            or (symbol.is_free() and symbol.get_name() not in enclosing_bound)
        )
    }
    supplied = enclosing_bound
    if table.get_type() not in {"module", "class"}:
        supplied = supplied.union(
            symbol.get_name() for symbol in table.get_symbols() if symbol.is_local()
        )
    elif table.get_type() == "class":
        supplied = supplied.union({"__class__", "__classdict__"})
    for child in table.get_children():
        dependencies.update(_symbol_table_dependencies(child, supplied))
    return dependencies


_OPAQUE_NAMESPACE_NAMES = {
    "globals",
    "locals",
    "vars",
    "eval",
    "exec",
    "__import__",
    "__builtins__",
}
_OPAQUE_NAMESPACE_ATTRIBUTES = {
    "__dict__",
    "__globals__",
    "__closure__",
    "__defaults__",
    "__kwdefaults__",
}


def _has_opaque_namespace_access(
    module: ast.AST, symbols: symtable.SymbolTable
) -> bool:
    """Detect explicit access to Python namespace capabilities.

    A scope that creates these capabilities can export them through arbitrary aliases
    or callables. Composed notebook code must therefore retain this scope as an opaque
    boundary instead of trying to infer the resulting Python objects.
    """
    if any(
        (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in _OPAQUE_NAMESPACE_NAMES
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name == "builtins" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and (
                node.module == "builtins"
                or any(alias.name == "*" for alias in node.names)
            )
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr in _OPAQUE_NAMESPACE_ATTRIBUTES
        )
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _OPAQUE_NAMESPACE_ATTRIBUTES
        )
        for node in ast.walk(module)
    ):
        return True
    declares_global = any(isinstance(node, ast.Global) for node in ast.walk(module))

    def deferred_writes(scope: symtable.SymbolTable) -> bool:
        return (
            (declares_global or scope.get_name() == "genexpr")
            and any(
                symbol.is_declared_global()
                and (symbol.is_assigned() or symbol.is_imported())
                for symbol in scope.get_symbols()
            )
        ) or any(deferred_writes(child) for child in scope.get_children())

    return any(deferred_writes(child) for child in symbols.get_children())


class _CurrentScopeNames(ast.NodeVisitor):
    """Collect lexical bindings and retain the existing statement-count view."""

    def __init__(
        self,
        local_names: set[str] | None = None,
        *,
        count_comprehension_locals: bool = False,
    ) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()
        self.writes: set[str] = set()
        self.load_counts: Counter[str] = Counter()
        self.store_counts: Counter[str] = Counter()
        self.definition_counts: Counter[str] = Counter()
        self.wildcard_stores = 0
        self._local_names = set(local_names or ())
        self._count_comprehension_locals = count_comprehension_locals

    def _store(self, name: str, *, counted: bool = True) -> None:
        if name not in self._local_names:
            self.stores.add(name)
            self.writes.add(name)
            if counted:
                self.store_counts[name] += 1

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self._local_names:
            return
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
            self.load_counts[node.id] += 1
        elif isinstance(node.ctx, ast.Store | ast.Del):
            self._store(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.visit(ast.Name(id=node.target.id, ctx=ast.Load()))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None and isinstance(node.target, ast.Name):
            self.stores.add(node.target.id)
            self.visit(node.annotation)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._store(alias.asname or alias.name.partition(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.wildcard_stores += node.module != "__future__"
            else:
                self._store(
                    alias.asname or alias.name, counted=node.module != "__future__"
                )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._store(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self._store(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._store(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self._store(node.rest)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._store(node.name, counted=False)
        self.definition_counts[node.name] += 1
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
        self._store(node.name, counted=False)
        self.definition_counts[node.name] += 1
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword_arg in node.keywords:
            self.visit(keyword_arg)

    def _visit_argument_expressions(self, args: ast.arguments) -> None:
        for default in (*args.defaults, *args.kw_defaults):
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

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
    ) -> None:
        if self._count_comprehension_locals:
            self.generic_visit(node)
            return
        previous_locals = self._local_names
        self._local_names = set(previous_locals)
        try:
            for generator in node.generators:
                self.visit(generator.iter)
                target = _CurrentScopeNames(self._local_names)
                target.visit(generator.target)
                self.loads.update(target.loads)
                self._local_names.update(target.stores)
                for condition in generator.ifs:
                    self.visit(condition)
            for field in ("elt", "key", "value"):
                if (value := getattr(node, field, None)) is not None:
                    self.visit(value)
        finally:
            self._local_names = previous_locals

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension
    visit_DictComp = _visit_comprehension


class _BindingState(typing.NamedTuple):
    """Separate possible availability, incoming-value replacement, and final binding."""

    available: frozenset[str] = frozenset()
    replaced: frozenset[str] = frozenset()
    deleted: frozenset[str] = frozenset()
    bound: frozenset[str] = frozenset()
    functions: Mapping[str, frozenset[str]] = {}


_BindingExit = typing.Literal["normal", "break", "continue", "return", "raise"]
_BindingFlow = dict[_BindingExit, _BindingState]


def _merge_binding_states(states: Sequence[_BindingState]) -> _BindingState:
    """Keep possible availability and effects shared by every path."""
    functions: dict[str, frozenset[str]] = {}
    for state in states:
        for name, dependencies in state.functions.items():
            functions[name] = functions.get(name, frozenset()) | dependencies
    return _BindingState(
        frozenset().union(*(state.available for state in states)),
        frozenset.intersection(*(state.replaced for state in states)),
        frozenset.intersection(*(state.deleted for state in states)),
        frozenset.intersection(*(state.bound for state in states)),
        functions,
    )


def _merge_binding_flows(*flows: _BindingFlow) -> _BindingFlow:
    exits: dict[_BindingExit, list[_BindingState]] = {}
    for flow in flows:
        for exit_kind, state in flow.items():
            exits.setdefault(exit_kind, []).append(state)
    return {kind: _merge_binding_states(states) for kind, states in exits.items()}


class _CodeBindingEffects:
    """Follow ordered module bindings without evaluating Python expressions.

    Availability retains the validator's permissive union of conditional bindings.
    Replacement and deletion are definite effects on normally completed paths. The
    same traversal supplies notebook liveness and optional name validation.
    """

    def __init__(
        self,
        *,
        read: Callable[[str, _BindingState], _BindingState] | None = None,
        require_binding: Callable[[str, _BindingState], _BindingState] | None = None,
        functions: Mapping[tuple[str, int], set[str]] | None = None,
    ) -> None:
        self.incoming_reads: set[str] = set()
        self.incoming_bindings: set[str] = set()
        self._read_callback = read
        self._require_binding_callback = require_binding
        self._functions = functions or {}

    def _read(self, name: str, state: _BindingState) -> _BindingState:
        if name not in state.replaced:
            self.incoming_reads.add(name)
        return (
            state if self._read_callback is None else self._read_callback(name, state)
        )

    @staticmethod
    def _bind(
        state: _BindingState, names: Collection[str], *, definite: bool = True
    ) -> _BindingState:
        names = frozenset(names)
        return _BindingState(
            state.available | names,
            state.replaced | names if definite else state.replaced,
            state.deleted - names,
            state.bound | names if definite else state.bound,
            {
                name: deps
                for name, deps in state.functions.items()
                if not definite or name not in names
            },
        )

    @staticmethod
    def _delete(state: _BindingState, names: Collection[str]) -> _BindingState:
        names = frozenset(names)
        return _BindingState(
            state.available - names,
            state.replaced | names,
            state.deleted | names,
            state.bound - names,
            {name: deps for name, deps in state.functions.items() if name not in names},
        )

    def _expression(self, node: ast.AST | None, state: _BindingState) -> _BindingState:
        if node is None:
            return state
        if isinstance(node, ast.Name):
            return (
                self._read(node.id, state) if isinstance(node.ctx, ast.Load) else state
            )
        if isinstance(node, ast.NamedExpr):
            return self._target(node.target, self._expression(node.value, state))
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                state = self._expression(default, state)
            return state
        if isinstance(
            node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
        ):
            names = _CurrentScopeNames()
            names.visit(node)
            for name in sorted(names.loads):
                state = self._read(name, state)
            return (
                state
                if isinstance(node, ast.GeneratorExp)
                else self._bind(state, names.writes, definite=False)
            )
        if isinstance(node, ast.IfExp):
            state = self._expression(node.test, state)
            return _merge_binding_states(
                [
                    self._expression(node.body, state),
                    self._expression(node.orelse, state),
                ]
            )
        if isinstance(node, ast.BoolOp | ast.Compare):
            values = (
                node.values
                if isinstance(node, ast.BoolOp)
                else [node.left, *node.comparators]
            )
            for index, value in enumerate(values):
                evaluated = self._expression(value, state)
                state = (
                    evaluated
                    if index < (1 if isinstance(node, ast.BoolOp) else 2)
                    else _merge_binding_states([state, evaluated])
                )
            return state
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                state = self._expression(key, state)
                state = self._expression(value, state)
            return state
        for child in ast.iter_child_nodes(node):
            state = self._expression(child, state)
        return state

    def _target(
        self, target: ast.AST, state: _BindingState, *, delete: bool = False
    ) -> _BindingState:
        if isinstance(target, ast.Name):
            if delete:
                if target.id not in state.replaced:
                    self.incoming_bindings.add(target.id)
                if self._require_binding_callback is not None:
                    state = self._require_binding_callback(target.id, state)
                return self._delete(state, (target.id,))
            return self._bind(state, (target.id,))
        if isinstance(target, ast.Tuple | ast.List | ast.Starred):
            for child in ast.iter_child_nodes(target):
                state = self._target(child, state, delete=delete)
            return state
        return self._expression(target, state)

    def block(
        self, statements: Sequence[ast.stmt], state: _BindingState | None = None
    ) -> _BindingFlow:
        flow: _BindingFlow = {"normal": state or _BindingState()}
        for statement in statements:
            normal = flow.pop("normal", None)
            if normal is None:
                break
            flow = _merge_binding_flows(flow, self._statement(statement, normal))
        return flow

    def _statement(self, node: ast.stmt, state: _BindingState) -> _BindingFlow:
        if isinstance(node, ast.If):
            state = self._expression(node.test, state)
            return _merge_binding_flows(
                self.block(node.body, state), self.block(node.orelse, state)
            )
        if isinstance(node, ast.For | ast.AsyncFor | ast.While):
            return self._loop(node, state)
        if isinstance(node, ast.Try | ast.TryStar):
            return self._try(node, state)
        if isinstance(node, ast.Match):
            state = self._expression(node.subject, state)
            branches: list[_BindingFlow] = []
            unmatched = True
            for case in node.cases:
                names = _CurrentScopeNames()
                names.visit(case.pattern)
                branch = state
                for name in sorted(names.loads):
                    branch = self._read(name, branch)
                branch = self._bind(branch, names.writes)
                branch = self._expression(case.guard, branch)
                branches.append(self.block(case.body, branch))
                if case.guard is None and self._irrefutable_pattern(case.pattern):
                    unmatched = False
            return _merge_binding_flows(
                *branches, {"normal": state} if unmatched else {}
            )
        if isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                state = self._expression(item.context_expr, state)
                if item.optional_vars is not None:
                    state = self._target(item.optional_vars, state)
            body = self.block(node.body, state)
            # A context manager can suppress an exception before any body binding.
            return _merge_binding_flows(body, {"normal": state})
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            if isinstance(node, ast.AugAssign):
                state = (
                    self._read(node.target.id, state)
                    if isinstance(node.target, ast.Name)
                    else self._expression(node.target, state)
                )
            state = self._expression(node.value, state)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not isinstance(node, ast.AnnAssign) or node.value is not None:
                for target in targets:
                    state = self._target(target, state)
            elif not isinstance(node.target, ast.Name):
                state = self._expression(node.target, state)
            if isinstance(node, ast.AnnAssign):
                state = self._expression(node.annotation, state)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                state = self._target(target, state, delete=True)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                state = self._expression(decorator, state)
            if isinstance(node, ast.ClassDef):
                for expression in (*node.bases, *node.keywords):
                    state = self._expression(expression, state)
            else:
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    state = self._expression(default, state)
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                    *(
                        arg
                        for arg in (node.args.vararg, node.args.kwarg)
                        if arg is not None
                    ),
                ):
                    state = self._expression(argument.annotation, state)
                state = self._expression(node.returns, state)
            state = self._bind(state, (node.name,))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                state = state._replace(
                    functions={
                        **state.functions,
                        node.name: frozenset(
                            self._functions.get((node.name, node.lineno), ())
                        ),
                    }
                )
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names = _CurrentScopeNames()
            names.visit(node)
            state = self._bind(state, names.writes)
        elif isinstance(node, ast.Assert):
            state = self._expression(node.test, state)
            return {"normal": state, "raise": self._expression(node.msg, state)}
        elif isinstance(node, ast.Return | ast.Raise):
            state = self._expression(node, state)
            return {"return" if isinstance(node, ast.Return) else "raise": state}
        elif isinstance(node, ast.Break | ast.Continue):
            return {"break" if isinstance(node, ast.Break) else "continue": state}
        else:
            state = self._expression(node, state)
            if not isinstance(node, ast.Expr):
                names = _CurrentScopeNames()
                names.visit(node)
                state = self._bind(state, names.writes, definite=False)
        return {"normal": state}

    @staticmethod
    def _irrefutable_pattern(pattern: ast.pattern) -> bool:
        return (
            isinstance(pattern, ast.MatchAs)
            and (
                pattern.pattern is None
                or _CodeBindingEffects._irrefutable_pattern(pattern.pattern)
            )
        ) or (
            isinstance(pattern, ast.MatchOr)
            and any(
                _CodeBindingEffects._irrefutable_pattern(p) for p in pattern.patterns
            )
        )

    def _loop(
        self, node: ast.For | ast.AsyncFor | ast.While, state: _BindingState
    ) -> _BindingFlow:
        state = self._expression(
            node.test if isinstance(node, ast.While) else node.iter, state
        )
        entry = (
            state if isinstance(node, ast.While) else self._target(node.target, state)
        )
        body = self.block(node.body, entry)
        exhausted = [state]
        iterations = [body.pop(kind) for kind in ("normal", "continue") if kind in body]
        exhausted.extend(
            self._expression(node.test, iteration)
            if isinstance(node, ast.While)
            else iteration
            for iteration in iterations
        )
        normal = self.block(node.orelse, _merge_binding_states(exhausted))
        broken = body.pop("break", None)
        return _merge_binding_flows(
            body, normal, {} if broken is None else {"normal": broken}
        )

    def _try(self, node: ast.Try | ast.TryStar, state: _BindingState) -> _BindingFlow:
        body = self.block(node.body, state)
        branches: list[_BindingFlow] = []
        for handler in node.handlers:
            # An exception can occur before the first binding in the try body.
            branch = self._expression(handler.type, state)
            if handler.name is not None:
                branch = self._bind(branch, (handler.name,))
            handled = self.block(handler.body, branch)
            if handler.name is not None:
                handled = {
                    kind: self._delete(value, (handler.name,))
                    for kind, value in handled.items()
                }
            branches.append(handled)
        normal = body.pop("normal", None)
        if normal is not None:
            branches.append(self.block(node.orelse, normal))
        combined = _merge_binding_flows(body, *branches)
        final: list[_BindingFlow] = []
        for kind, value in combined.items():
            after = self.block(node.finalbody, value)
            completed = after.pop("normal", None)
            final.append(
                _merge_binding_flows(
                    after, {} if completed is None else {kind: completed}
                )
            )
        return _merge_binding_flows(*final)


class _CodeAnalysis(typing.NamedTuple):
    module: ast.Module
    symbols: symtable.SymbolTable
    loads: frozenset[str]
    writes: frozenset[str]
    deletions: frozenset[str]
    incoming_reads: frozenset[str]
    incoming_bindings: frozenset[str]
    definite_writes: frozenset[str]
    definite_bindings: frozenset[str]
    deferred_reads: frozenset[str]
    referenced_names: frozenset[str]
    opaque: bool


def _incoming_binding_names(statements: Sequence[ast.stmt]) -> set[str]:
    """Return incoming bindings needed without reading their values."""
    effects = _CodeBindingEffects()
    effects.block(statements)
    return effects.incoming_bindings


class _CodeAnalyzer:
    """Cache read-only analysis facts for one code-generation request."""

    def __init__(self) -> None:
        self._analyses: dict[str, _CodeAnalysis] = {}

    def __call__(self, code: str) -> _CodeAnalysis:
        if code not in self._analyses:
            module = ast.parse(code, mode="exec")
            symbols = symtable.symtable(code, "<ImageTool replay scope>", "exec")
            names = _CurrentScopeNames()
            names.visit(module)
            effects = _CodeBindingEffects()
            final = effects.block(module.body).get("normal", _BindingState())
            self._analyses[code] = _CodeAnalysis(
                module,
                symbols,
                frozenset(names.loads),
                frozenset(names.writes),
                final.deleted,
                frozenset(effects.incoming_reads),
                frozenset(effects.incoming_bindings),
                final.replaced,
                final.bound,
                frozenset().union(
                    *(
                        _symbol_table_dependencies(child)
                        for child in symbols.get_children()
                    )
                ),
                frozenset(_symbol_table_dependencies(symbols)),
                _has_opaque_namespace_access(module, symbols),
            )
        return self._analyses[code]


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
    names = _CurrentScopeNames(count_comprehension_locals=True)
    names.visit(stmt)
    return names.load_counts[target]


def _statement_store_count(
    stmt: ast.stmt, target: str, *, count_definition_names: bool = False
) -> int:
    names = _CurrentScopeNames(count_comprehension_locals=True)
    names.visit(stmt)
    return (
        names.store_counts[target]
        + names.wildcard_stores
        + (names.definition_counts[target] if count_definition_names else 0)
    )


class _NameReplacer(_ScopedNameReplacer):
    def __init__(self, target: str, replacement: ast.expr) -> None:
        super().__init__(target, replacement)


def _clone_expr(node: ast.expr) -> ast.expr:
    return ast.parse(ast.unparse(node), mode="eval").body


def _clone_stmt(node: ast.stmt) -> ast.stmt:
    return ast.parse(ast.unparse(node), mode="exec").body[0]


class _ScopeAwareTransformer(ast.NodeTransformer):
    """Traverse an AST using Python's own lexical-scope analysis."""

    def __init__(self, code: str) -> None:
        self._scope = symtable.symtable(code, "<provenance>", "exec")
        self._used_scope_children: dict[int, set[int]] = {}
        self._scope_children: dict[int, tuple[symtable.SymbolTable, ...]] = {}
        self._scope_parents: dict[int, symtable.SymbolTable] = {}
        self._resolution_scope: symtable.SymbolTable | None = None
        self._hidden_bindings: set[str] = set()
        self.generated_import_relay_ids: set[int] = set()

    def _resolves_to_module(self, name: str) -> bool:
        if name in self._hidden_bindings:
            return False
        scope = self._resolution_scope or self._scope
        if scope.get_type() == "module":
            return True
        try:
            return scope.lookup(name).is_global()
        except KeyError:
            return True

    def _rewrite_binding(self, name: str) -> str:
        return name

    def _visit_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        visited: list[ast.stmt] = []
        for statement in statements:
            result = self.visit(statement)
            if result is None:
                continue
            if isinstance(result, list):
                visited.extend(typing.cast("list[ast.stmt]", result))
            else:
                visited.append(typing.cast("ast.stmt", result))
        return visited

    def _visit_type_parameters(self, node: ast.AST) -> set[str]:
        previous_hidden = self._hidden_bindings
        type_parameters = list(getattr(node, "type_params", ()))
        self._hidden_bindings = previous_hidden | {
            parameter.name
            for parameter in type_parameters
            if isinstance(getattr(parameter, "name", None), str)
        }
        if hasattr(node, "type_params"):
            node.type_params = [self.visit(parameter) for parameter in type_parameters]
        return previous_hidden

    def _find_child_scope(
        self,
        *,
        scope_type: str,
        name: str,
        lineno: int,
    ) -> symtable.SymbolTable | None:
        def find_in(parent: symtable.SymbolTable) -> symtable.SymbolTable | None:
            used = self._used_scope_children.setdefault(id(parent), set())
            children = self._scope_children.setdefault(
                id(parent), tuple(parent.get_children())
            )
            for child in children:
                self._scope_parents.setdefault(id(child), parent)
                if (
                    id(child) not in used
                    and child.get_type() == scope_type
                    and child.get_name() == name
                    and child.get_lineno() == lineno
                ):
                    used.add(id(child))
                    return child
            for child in children:
                if child.get_type() not in {
                    "annotation",
                    "type alias",
                    "type parameter",
                    "type parameters",
                }:
                    continue
                if found := find_in(child):
                    return found
            return None

        if found := find_in(self._scope):
            return found
        return None

    def _enter_scope(
        self,
        *,
        scope_type: str,
        name: str,
        lineno: int,
    ) -> tuple[symtable.SymbolTable, symtable.SymbolTable | None]:
        previous = self._scope, self._resolution_scope
        child = self._find_child_scope(
            scope_type=scope_type,
            name=name,
            lineno=lineno,
        )
        if child is None:
            raise RuntimeError(f"Could not resolve generated-code scope {name!r}")
        self._scope = child
        self._resolution_scope = None
        return previous

    def visit_Import(self, node: ast.Import) -> ast.AST | list[ast.stmt]:
        root_aliases: list[ast.Assign] = []
        relayed_roots: set[tuple[str, str]] = set()
        for alias in node.names:
            bound_name = alias.asname or alias.name.partition(".")[0]
            replacement = self._rewrite_binding(bound_name)
            if replacement == bound_name:
                continue
            if alias.asname is None and "." in alias.name:
                relay_key = (replacement, bound_name)
                if relay_key in relayed_roots:
                    continue
                relayed_roots.add(relay_key)
                relay = ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=replacement, ctx=ast.Store())],
                        value=ast.Name(id=bound_name, ctx=ast.Load()),
                    ),
                    node,
                )
                root_aliases.append(relay)
                self.generated_import_relay_ids.add(id(relay))
                continue
            alias.asname = replacement
        if root_aliases:
            return [node, *root_aliases]
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        for alias in node.names:
            if alias.name == "*":
                continue
            bound_name = alias.asname or alias.name
            replacement = self._rewrite_binding(bound_name)
            if replacement != bound_name:
                alias.asname = replacement
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self._rewrite_binding(name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name is not None:
            node.name = self._rewrite_binding(node.name)
        return self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        if node.name is not None:
            node.name = self._rewrite_binding(node.name)
        return self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        if node.name is not None:
            node.name = self._rewrite_binding(node.name)
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        if node.rest is not None:
            node.rest = self._rewrite_binding(node.rest)
        return self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        original_name = node.name
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        previous_hidden = self._visit_type_parameters(node)
        _visit_argument_expressions_with_transformer(node.args, self)
        if node.returns is not None:
            node.returns = typing.cast("ast.expr", self.visit(node.returns))
        self._hidden_bindings = previous_hidden
        node.name = self._rewrite_binding(node.name)
        previous = self._enter_scope(
            scope_type="function",
            name=original_name,
            lineno=node.lineno,
        )
        node.body = self._visit_statements(node.body)
        self._scope, self._resolution_scope = previous
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        _visit_argument_expressions_with_transformer(node.args, self)
        previous = self._enter_scope(
            scope_type="function",
            name="lambda",
            lineno=node.lineno,
        )
        node.body = typing.cast("ast.expr", self.visit(node.body))
        self._scope, self._resolution_scope = previous
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        original_name = node.name
        node.decorator_list = [
            typing.cast("ast.expr", self.visit(decorator))
            for decorator in node.decorator_list
        ]
        previous_hidden = self._visit_type_parameters(node)
        node.bases = [typing.cast("ast.expr", self.visit(base)) for base in node.bases]
        node.keywords = [
            typing.cast("ast.keyword", self.visit(keyword_arg))
            for keyword_arg in node.keywords
        ]
        self._hidden_bindings = previous_hidden
        node.name = self._rewrite_binding(node.name)
        previous = self._enter_scope(
            scope_type="class",
            name=original_name,
            lineno=node.lineno,
        )
        node.body = self._visit_statements(node.body)
        self._scope, self._resolution_scope = previous
        return node

    def visit_TypeAlias(self, node: ast.AST) -> ast.AST:
        type_alias = typing.cast("typing.Any", node)
        name_node = typing.cast("ast.Name", type_alias.name)
        original_name = name_node.id
        type_alias.name = typing.cast("ast.Name", self.visit(name_node))
        previous_hidden = self._visit_type_parameters(type_alias)
        previous = self._enter_scope(
            scope_type="type alias",
            name=original_name,
            lineno=type_alias.lineno,
        )
        type_alias.value = typing.cast("ast.expr", self.visit(type_alias.value))
        self._scope, self._resolution_scope = previous
        self._hidden_bindings = previous_hidden
        return type_alias

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> ast.AST:
        names = {
            ast.ListComp: "listcomp",
            ast.SetComp: "setcomp",
            ast.DictComp: "dictcomp",
            ast.GeneratorExp: "genexpr",
        }
        first_generator = node.generators[0]
        first_generator.iter = typing.cast("ast.expr", self.visit(first_generator.iter))
        child_scope = self._find_child_scope(
            scope_type="function", name=names[type(node)], lineno=node.lineno
        )
        previous_scope = self._scope
        previous_resolution_scope = self._resolution_scope
        previous_hidden_bindings = self._hidden_bindings
        if child_scope is None:
            if self._scope.get_type() == "class":
                self._resolution_scope = self._scope_parents[id(self._scope)]
            self._hidden_bindings = self._hidden_bindings | {
                child.id
                for generator in node.generators
                for child in ast.walk(generator.target)
                if isinstance(child, ast.Name)
            }
        else:
            self._scope = child_scope
            self._resolution_scope = None
        for index, generator in enumerate(node.generators):
            if index:
                generator.iter = typing.cast("ast.expr", self.visit(generator.iter))
            generator.target = typing.cast("ast.expr", self.visit(generator.target))
            generator.ifs = [
                typing.cast("ast.expr", self.visit(condition))
                for condition in generator.ifs
            ]
        if isinstance(node, ast.DictComp):
            node.key = typing.cast("ast.expr", self.visit(node.key))
            node.value = typing.cast("ast.expr", self.visit(node.value))
        else:
            node.elt = typing.cast("ast.expr", self.visit(node.elt))
        self._scope = previous_scope
        self._resolution_scope = previous_resolution_scope
        self._hidden_bindings = previous_hidden_bindings
        return node

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension


class _ModuleNameLoadCounter(_ScopeAwareTransformer):
    def __init__(self, code: str, target: str) -> None:
        super().__init__(code)
        self._target = target
        self.count = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            node.id == self._target
            and isinstance(node.ctx, ast.Load)
            and self._resolves_to_module(node.id)
        ):
            self.count += 1
        return node


class _IdentifierReplacer(_ScopeAwareTransformer):
    def __init__(self, code: str, replacements: Mapping[str, str]) -> None:
        super().__init__(code)
        self._replacements = dict(replacements)

    def _rewrite_binding(self, name: str) -> str:
        if not self._resolves_to_module(name):
            return name
        replacement = self._replacements.get(name, name)
        if replacement != name and not self._resolves_to_module(replacement):
            raise ValueError(
                f"Replacement identifier {replacement!r} is shadowed in a nested scope"
            )
        return replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self._rewrite_binding(node.id)
        if replacement == node.id:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


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


def _remove_unused_generated_import_relays(
    module: ast.Module,
    relay_ids: set[int],
) -> None:
    """Remove dotted-import aliases overwritten before they can be observed."""
    if not relay_ids:
        return

    def has_observable_use_in_any_scope(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load | ast.Del)
            and child.id == name
            for child in ast.walk(node)
        )

    def definitely_assigns_name(statement: ast.stmt, name: str) -> bool:
        if isinstance(statement, ast.Assign):
            return any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            )
        return (
            isinstance(statement, ast.AnnAssign)
            and statement.value is not None
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        )

    def clean_statement_list(statements: list[ast.stmt]) -> None:
        for index in range(len(statements) - 1, -1, -1):
            statement = statements[index]
            if id(statement) not in relay_ids:
                continue
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                continue  # pragma: no cover - relay construction is fixed above.
            target_name = statement.targets[0].id
            for later in statements[index + 1 :]:
                if has_observable_use_in_any_scope(later, target_name):
                    break
                if definitely_assigns_name(later, target_name):
                    del statements[index]
                    break

    def visit_statement_lists(node: ast.AST) -> None:
        for _field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit_statement_lists(value)
                continue
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, ast.AST):
                    visit_statement_lists(item)
            if value and all(isinstance(item, ast.stmt) for item in value):
                clean_statement_list(typing.cast("list[ast.stmt]", value))

    visit_statement_lists(module)


def _replace_code_identifiers(code: str, replacements: Mapping[str, str]) -> str:
    module = ast.parse(code, mode="exec")
    replacer = _IdentifierReplacer(code, replacements)
    replaced = typing.cast(
        "ast.Module",
        replacer.visit(module),
    )
    _remove_unused_generated_import_relays(
        replaced,
        replacer.generated_import_relay_ids,
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
    if current_name is None:
        assigned_names: set[str] = set()
        for code in codes:
            try:
                module = ast.parse(code, mode="exec")
            except SyntaxError:
                return None
            if not module.body:
                return None
            for statement in module.body:
                if (
                    not isinstance(statement, ast.Assign)
                    or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                ):
                    return None
                assigned_names.add(statement.targets[0].id)
        if len(assigned_names) == 1:
            return assigned_names.pop()
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
                and isinstance(stmt.value, ast.Name)
                and target_expr.id not in _incoming_binding_names(body[idx + 1 :])
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
            later_loads = [
                later_idx
                for later_idx, later in enumerate(body[idx + 1 :], start=idx + 1)
                if _statement_load_count(later, target) > 0
            ]
            if len(later_loads) != 1:
                continue

            next_idx = later_loads[0]
            next_stmt = body[next_idx]
            if (
                next_idx != idx + 1
                or not isinstance(next_stmt, ast.Assign)
                or not _expression_starts_with_name(next_stmt.value, target)
            ):
                continue
            reassigns_target_immediately = (
                len(next_stmt.targets) == 1
                and isinstance(next_stmt.targets[0], ast.Name)
                and next_stmt.targets[0].id == target
            )
            if (
                inline_targets is not None
                and target not in inline_targets
                and not reassigns_target_immediately
            ):
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
                    or _statement_load_count(intervening, name)
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


def _expression_starts_with_name(expression: ast.expr, name: str) -> bool:
    """Return whether evaluating an expression starts by loading ``name``.

    Generated-code cleanup may inline an adjacent assignment only at this position.
    That preserves evaluation order even when either expression has side effects.
    """
    while True:
        if isinstance(expression, ast.Name):
            return expression.id == name
        if isinstance(expression, ast.Attribute | ast.Subscript):
            expression = expression.value
            continue
        if isinstance(expression, ast.Call):
            expression = expression.func
            continue
        if isinstance(expression, ast.BinOp):
            expression = expression.left
            continue
        if isinstance(expression, ast.BoolOp):
            expression = expression.values[0]
            continue
        if isinstance(expression, ast.Compare):
            expression = expression.left
            continue
        if isinstance(expression, ast.UnaryOp | ast.NamedExpr):
            expression = (
                expression.operand
                if isinstance(expression, ast.UnaryOp)
                else expression.value
            )
            continue
        return False


def _code_uses_name(code: str, name: str) -> bool:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    counter = _ModuleNameLoadCounter(code, name)
    counter.visit(module)
    return counter.count > 0


def _code_uses_name_any_scope(code: str, name: str) -> bool:
    """Return whether code reads a name or needs its incoming module binding."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    return name in _incoming_binding_names(module.body) or any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(module)
    )


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


def _migrate_legacy_nonuniform_restore_code(code: str | None) -> str | None:
    """Rewrite the removed public restore helper in persisted executable code."""
    if code is None or "restore_nonuniform_dims" not in code:
        return code
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    encoded = code.encode()
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    replacements: list[tuple[int, int]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Attribute):
            continue
        if _receiver_path(node) != _LEGACY_NONUNIFORM_RESTORE_PATH:
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            continue  # pragma: no cover - populated by supported Python versions
        start = line_starts[node.lineno - 1] + node.col_offset
        end = line_starts[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end))

    if not replacements:
        return code

    replacement = _PRIVATE_NONUNIFORM_RESTORE_EXPRESSION.encode()
    for start, end in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode()


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


def _parse_validated_script_replay_code(code: str) -> ast.Module:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Script replay code is not valid Python: {exc}") from exc

    def validate_import(node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.ImportFrom):
            raise TypeError("Script replay code contains unsupported ImportFrom")
        if any(
            alias.name not in ("erlab", "lmfit", "numpy", "xarray")
            or (alias.asname is not None and alias.asname.startswith("__"))
            for alias in node.names
        ):
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

    def validate_numeric_conversion_try(node: ast.Try) -> None:
        if (
            node.orelse
            or node.finalbody
            or len(node.body) != 1
            or len(node.handlers) != 1
        ):
            raise TypeError("Script replay code contains unsupported Try")
        statement = node.body[0]
        handler = node.handlers[0]
        exception_names = (
            tuple(item.id for item in handler.type.elts if isinstance(item, ast.Name))
            if isinstance(handler.type, ast.Tuple)
            else ((handler.type.id,) if isinstance(handler.type, ast.Name) else ())
        )
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(statement.value, ast.Call)
            or not isinstance(statement.value.func, ast.Attribute)
            or statement.value.func.attr != "astype"
            or len(statement.value.args) != 1
            or statement.value.keywords
            or exception_names != ("TypeError", "ValueError")
            or handler.name is not None
            or len(handler.body) != 1
            or not isinstance(handler.body[0], ast.Continue)
        ):
            raise TypeError("Script replay code contains unsupported Try")

    def validate_safe_try(node: ast.Try) -> None:
        try:
            validate_optional_import_try(node)
        except TypeError:
            validate_numeric_conversion_try(node)

    for node in ast.walk(module):
        if isinstance(node, ast.Import | ast.ImportFrom):
            validate_import(node)
            continue
        if isinstance(node, ast.Try):
            validate_safe_try(node)
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

    return module


def _validate_script_replay_code(code: str) -> None:
    _parse_validated_script_replay_code(code)


def _script_replay_import_names(code: str) -> frozenset[str]:
    """Return top-level modules imported by validated replay code."""
    module = _parse_validated_script_replay_code(code)
    return frozenset(
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    )


class _ScriptReplayImportLowerer(ast.NodeTransformer):
    """Replace approved imports with assignments to executor-owned bindings."""

    def visit_Import(self, node: ast.Import) -> list[ast.Assign]:
        assignments: list[ast.Assign] = []
        for alias in node.names:
            if alias.name == "erlab":
                binding_name = "__erlab_replay_import_erlab"
            elif alias.name == "lmfit":
                binding_name = "__erlab_replay_import_lmfit"
            elif alias.name == "numpy":
                binding_name = "__erlab_replay_import_numpy"
            elif alias.name == "xarray":
                binding_name = "__erlab_replay_import_xarray"
            else:  # pragma: no cover - validation rejects every other import
                raise TypeError("Script replay code contains unsupported Import")
            assignments.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[
                            ast.Name(
                                id=alias.asname or alias.name,
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Name(id=binding_name, ctx=ast.Load()),
                    ),
                    node,
                )
            )
        return assignments


def _compile_untrusted_script_replay_code(code: str) -> types.CodeType:
    """Validate and compile replay code without executing Python imports."""
    module = _parse_validated_script_replay_code(code)
    lowered = typing.cast("ast.Module", _ScriptReplayImportLowerer().visit(module))
    ast.fix_missing_locations(lowered)
    return compile(lowered, "<ImageTool script provenance>", "exec")
