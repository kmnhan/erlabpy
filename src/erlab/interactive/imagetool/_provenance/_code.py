"""Code generation and restricted-script policy for ImageTool provenance."""

from __future__ import annotations

import ast
import keyword
import math
import symtable
import typing
from collections.abc import Hashable, Mapping, Sequence

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

    def visit_Import(self, node: ast.Import) -> None:
        if ast.Store not in self.contexts:
            return
        for alias in node.names:
            bound_name = alias.asname or alias.name.partition(".")[0]
            self.count += bound_name == self.target

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if ast.Store not in self.contexts or node.module == "__future__":
            return
        for alias in node.names:
            if alias.name == "*":
                self.count += 1
                continue
            self.count += (alias.asname or alias.name) == self.target

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if ast.Store in self.contexts and node.name is not None:
            self.count += node.name == self.target
        if node.type is not None:
            self.visit(node.type)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if ast.Store in self.contexts and node.name is not None:
            self.count += node.name == self.target
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if ast.Store in self.contexts and node.name is not None:
            self.count += node.name == self.target

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if ast.Store in self.contexts and node.rest is not None:
            self.count += node.rest == self.target
        for key in node.keys:
            self.visit(key)
        for pattern in node.patterns:
            self.visit(pattern)

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


class _ModuleNameReplacer(_ScopeAwareTransformer):
    def __init__(self, code: str, target: str, replacement: ast.expr) -> None:
        super().__init__(code)
        self._target = target
        self._replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            node.id == self._target
            and isinstance(node.ctx, ast.Load)
            and self._resolves_to_module(node.id)
        ):
            return _clone_expr(self._replacement)
        return node


class _ModuleNameLoadCounter(_ModuleNameReplacer):
    def __init__(self, code: str, target: str) -> None:
        super().__init__(code, target, ast.Name(id=target, ctx=ast.Load()))
        self.count = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            node.id == self._target
            and isinstance(node.ctx, ast.Load)
            and self._resolves_to_module(node.id)
        ):
            self.count += 1
        return node


class _ModuleNameReplacementCollisionDetector(_ScopeAwareTransformer):
    def __init__(self, code: str, target: str, replacement_names: set[str]) -> None:
        super().__init__(code)
        self._target = target
        self._replacement_names = replacement_names
        self.collision = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            node.id == self._target
            and isinstance(node.ctx, ast.Load)
            and self._resolves_to_module(node.id)
            and any(
                not self._resolves_to_module(name) for name in self._replacement_names
            )
        ):
            self.collision = True
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
    """Return whether code loads a name at module or nested scope."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(module)
    )


def _scope_identifiers(table: symtable.SymbolTable) -> set[str]:
    identifiers = set(table.get_identifiers())
    for child in table.get_children():
        identifiers.update(_scope_identifiers(child))
    return identifiers


def _collision_free_source_name(code: str, input_expr: ast.expr) -> str:
    unavailable = _scope_identifiers(symtable.symtable(code, "<provenance>", "exec"))
    unavailable.update(
        node.id for node in ast.walk(input_expr) if isinstance(node, ast.Name)
    )
    name = "source_data"
    suffix = 2
    while name in unavailable:
        name = f"source_data_{suffix}"
        suffix += 1
    return name


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

    replacement_names = {
        node.id
        for node in ast.walk(input_expr)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    collision_detector = _ModuleNameReplacementCollisionDetector(
        code,
        "data",
        replacement_names,
    )
    collision_detector.visit(module)
    source_assignment: ast.Assign | None = None
    if collision_detector.collision:
        source_name = _collision_free_source_name(code, input_expr)
        source_assignment = ast.Assign(
            targets=[ast.Name(id=source_name, ctx=ast.Store())],
            value=input_expr,
        )
        input_expr = ast.Name(id=source_name, ctx=ast.Load())

    replacer = _ModuleNameReplacer(code, "data", input_expr)
    rebased = typing.cast("ast.Module", replacer.visit(module))
    _remove_unused_generated_import_relays(
        rebased,
        replacer.generated_import_relay_ids,
    )
    if source_assignment is not None:
        insert_at = 0
        if (
            rebased.body
            and isinstance(rebased.body[0], ast.Expr)
            and isinstance(rebased.body[0].value, ast.Constant)
            and isinstance(rebased.body[0].value.value, str)
        ):
            insert_at = 1
        while insert_at < len(rebased.body):
            statement = rebased.body[insert_at]
            if not (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
            ):
                break
            insert_at += 1
        rebased.body.insert(insert_at, source_assignment)
    rebased = ast.fix_missing_locations(rebased)
    return _simplify_display_code(
        ast.unparse(rebased),
        inline_targets={"derived"},
    )


def uses_default_replay_input(code: str) -> bool:
    """Return whether generated replay code refers to the generic ``data`` input."""
    return _code_uses_name(code, "data")


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
