"""Jupyter console widget for ImageToolManager."""

from __future__ import annotations

__all__ = ["ToolNamespace", "ToolsNamespace", "_ImageToolManagerJupyterConsole"]

import ast
import contextlib
import functools
import importlib
import inspect
import keyword
import operator
import symtable
import textwrap
import types
import typing
import weakref
from dataclasses import dataclass

import numpy as np
import qtconsole.inprocess
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.utils
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from IPython.core.interactiveshell import (
        ExecutionInfo,
        ExecutionResult,
        InteractiveShell,
    )

    from erlab.interactive.imagetool import ImageTool
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


def _patch_macos_matplotlib_qt_cursor() -> None:
    """Disable Matplotlib Qt cursor updates on macOS."""
    erlab.interactive.utils.patch_macos_matplotlib_qt_cursor()


def _resolve_console_namespace(
    namespace: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    _patch_macos_matplotlib_qt_cursor()
    resolved = {}
    for name, module in namespace.items():
        value = importlib.import_module(module) if isinstance(module, str) else module
        if name in {"erlab", "era", "eri", "xr"} and isinstance(
            value, types.ModuleType
        ):
            value = _ConsoleModuleProxy(value, name)
        resolved[name] = value
    return resolved


def _tool_data_name(index: int) -> str:
    return f"data_{index}"


def _dedupe_script_inputs(
    inputs: Sequence[provenance.ScriptInput],
) -> tuple[provenance.ScriptInput, ...]:
    deduped: list[provenance.ScriptInput] = []
    seen: set[str] = set()
    for script_input in inputs:
        if script_input.name in seen:
            continue
        seen.add(script_input.name)
        deduped.append(script_input)
    return tuple(deduped)


def _dedupe_code_preludes(*groups: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for code in group:
            if code in seen:
                continue
            seen.add(code)
            output.append(code)
    return tuple(output)


def _script_code(prelude: Sequence[str], code: str) -> str:
    return "\n\n".join((*prelude, code)) if prelude else code


def _replay_name_reserved(name: str) -> bool:
    return name in {"data", "derived", "tools"} or name.startswith("data_")


def _function_global_names(source: str, function_name: str) -> set[str] | None:
    try:
        table = symtable.symtable(source, "<ImageTool console function>", "exec")
    except SyntaxError:
        return None

    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.get_name() != function_name
        and symbol.is_referenced()
        and symbol.is_global()
    }
    for child in table.get_children():
        if child.get_name() != function_name:
            continue
        names.update(
            symbol.get_name()
            for symbol in child.get_symbols()
            if symbol.get_name() != function_name
            and symbol.is_referenced()
            and (symbol.is_global() or symbol.is_free())
        )
        return names
    return None


def _callable_operand(
    value: typing.Any, seen: set[int] | None = None
) -> _ConsoleOperand | None:
    if getattr(value, "_erlab_console_function_proxy", False):
        value = value.__wrapped__
    if (
        not inspect.isfunction(value)
        or value.__module__ != "__main__"
        or value.__qualname__ != value.__name__
        or value.__closure__ is not None
    ):
        return None

    name = value.__name__
    if (
        not name.isidentifier()
        or keyword.iskeyword(name)
        or _replay_name_reserved(name)
    ):
        return None
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return None
    seen.add(value_id)

    try:
        source = textwrap.dedent(inspect.getsource(value)).strip()
        module = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    function_node = module.body[0]
    if function_node.name != name or function_node.decorator_list:
        return None

    global_names = _function_global_names(source, name)
    if global_names is None:
        return None

    prelude: list[str] = []
    allowed_globals = {
        "np",
        "numpy",
        "xr",
        "xarray",
        "erlab",
        "era",
        "eri",
        "eplt",
        *provenance._SCRIPT_REPLAY_ALLOWED_BUILTINS,
    }
    for global_name in sorted(global_names):
        if global_name in allowed_globals:
            continue
        if _replay_name_reserved(global_name) or global_name.startswith("__"):
            return None
        try:
            global_value = value.__globals__[global_name]
        except KeyError:
            return None
        if inspect.isfunction(global_value) or getattr(
            global_value, "_erlab_console_function_proxy", False
        ):
            dependency = _callable_operand(global_value, seen)
            if dependency is None:
                return None
            prelude.extend(dependency.code_prelude)
            if dependency.code != global_name:
                prelude.append(f"{global_name} = {dependency.code}")
            continue
        global_code, copyable = _literal_code(global_value)
        if not copyable:
            return None
        prelude.append(f"{global_name} = {global_code}")

    code_prelude = _dedupe_code_preludes(prelude, (source,))
    try:
        provenance._validate_script_replay_code(
            _script_code(code_prelude, "derived = data")
        )
    except (TypeError, ValueError):
        return None
    return _ConsoleOperand(value, name, copyable=True, code_prelude=code_prelude)


@dataclass(frozen=True)
class _ConsoleOperand:
    value: typing.Any
    code: str
    script_inputs: tuple[provenance.ScriptInput, ...] = ()
    copyable: bool = True
    code_prelude: tuple[str, ...] = ()
    seed_expression: str | None = None
    operations: tuple[provenance.ToolProvenanceOperation, ...] = ()


def _literal_code(value: typing.Any) -> tuple[str, bool]:
    value = erlab.utils.misc._convert_to_native(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "np.nan", True
        if np.isinf(value):
            return ("np.inf" if value > 0 else "-np.inf"), True
        return repr(value), True
    if isinstance(value, complex):
        real_code, _ = _literal_code(float(value.real))
        imag_code, _ = _literal_code(float(value.imag))
        return f"complex({real_code}, {imag_code})", True
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value), True
    if isinstance(value, slice):
        start, start_copyable = _literal_code(value.start)
        stop, stop_copyable = _literal_code(value.stop)
        step, step_copyable = _literal_code(value.step)
        return (
            f"slice({start}, {stop}, {step})",
            start_copyable and stop_copyable and step_copyable,
        )
    if isinstance(value, tuple):
        parts = [_literal_code(item) for item in value]
        suffix = "," if len(parts) == 1 else ""
        return (
            f"({', '.join(part for part, _copyable in parts)}{suffix})",
            all(copyable for _part, copyable in parts),
        )
    if isinstance(value, list):
        parts = [_literal_code(item) for item in value]
        return (
            f"[{', '.join(part for part, _copyable in parts)}]",
            all(copyable for _part, copyable in parts),
        )
    if isinstance(value, dict):
        dict_parts: list[str] = []
        copyable = True
        for key, item in value.items():
            key_code, key_copyable = _literal_code(key)
            item_code, item_copyable = _literal_code(item)
            dict_parts.append(f"{key_code}: {item_code}")
            copyable = copyable and key_copyable and item_copyable
        return f"{{{', '.join(dict_parts)}}}", copyable
    return repr(value), False


def _merge_operands(
    *operands: _ConsoleOperand,
) -> tuple[
    tuple[provenance.ScriptInput, ...],
    bool,
    tuple[str, ...],
]:
    return (
        _dedupe_script_inputs(
            tuple(
                script_input
                for operand in operands
                for script_input in operand.script_inputs
            )
        ),
        all(operand.copyable for operand in operands),
        _dedupe_code_preludes(*(operand.code_prelude for operand in operands)),
    )


def _derived_operand_code(expression: str) -> str:
    if any(symbol in expression for symbol in (" + ", " - ", " * ", " / ", " // ")):
        return f"({expression})"
    if any(symbol in expression for symbol in (" % ", " ** ", " & ", " | ", " ^ ")):
        return f"({expression})"
    if any(
        symbol in expression
        for symbol in (" < ", " <= ", " > ", " >= ", " == ", " != ")
    ):
        return f"({expression})"
    return expression


def _unwrap_console_value(value: typing.Any) -> typing.Any:
    if isinstance(value, _ConsoleDataHandleBase):
        return value.data
    if isinstance(value, tuple):
        return tuple(_unwrap_console_value(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_console_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _unwrap_console_value(key): _unwrap_console_value(item)
            for key, item in value.items()
        }
    return value


def _first_console_handle(value: typing.Any) -> _ConsoleDataHandleBase | None:
    if isinstance(value, _ConsoleDataHandleBase):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if (handle := _first_console_handle(item)) is not None:
                return handle
    if isinstance(value, dict):
        for item in (*value.keys(), *value.values()):
            if (handle := _first_console_handle(item)) is not None:
                return handle
    return None


def _operand_from_value(value: typing.Any) -> _ConsoleOperand:
    if isinstance(value, _ConsoleDataHandleBase):
        return value._console_operand()
    if (callable_operand := _callable_operand(value)) is not None:
        return callable_operand
    if isinstance(value, tuple):
        item_operands = tuple(_operand_from_value(item) for item in value)
        suffix = "," if len(item_operands) == 1 else ""
        inputs, copyable, code_prelude = _merge_operands(*item_operands)
        return _ConsoleOperand(
            tuple(operand.value for operand in item_operands),
            f"({', '.join(operand.code for operand in item_operands)}{suffix})",
            inputs,
            copyable,
            code_prelude,
        )
    if isinstance(value, list):
        item_operands = tuple(_operand_from_value(item) for item in value)
        inputs, copyable, code_prelude = _merge_operands(*item_operands)
        return _ConsoleOperand(
            [operand.value for operand in item_operands],
            f"[{', '.join(operand.code for operand in item_operands)}]",
            inputs,
            copyable,
            code_prelude,
        )
    if isinstance(value, dict):
        key_operands = tuple(_operand_from_value(key) for key in value)
        value_operands = tuple(_operand_from_value(item) for item in value.values())
        pair_operands = tuple(
            operand
            for pair in zip(key_operands, value_operands, strict=True)
            for operand in pair
        )
        inputs, copyable, code_prelude = _merge_operands(*pair_operands)
        return _ConsoleOperand(
            {
                key_operand.value: value_operand.value
                for key_operand, value_operand in zip(
                    key_operands, value_operands, strict=True
                )
            },
            "{"
            + ", ".join(
                f"{key_operand.code}: {value_operand.code}"
                for key_operand, value_operand in zip(
                    key_operands, value_operands, strict=True
                )
            )
            + "}",
            inputs,
            copyable,
            code_prelude,
        )
    code, copyable = _literal_code(value)
    return _ConsoleOperand(value, code, copyable=copyable)


def _format_call_code(
    args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]
) -> tuple[
    str,
    tuple[provenance.ScriptInput, ...],
    tuple[typing.Any, ...],
    dict[str, typing.Any],
    bool,
    tuple[str, ...],
]:
    arg_operands = tuple(_operand_from_value(arg) for arg in args)
    kwarg_operands = {key: _operand_from_value(value) for key, value in kwargs.items()}
    inputs, copyable, code_prelude = _merge_operands(
        *arg_operands, *tuple(kwarg_operands.values())
    )
    parts = [operand.code for operand in arg_operands]
    parts.extend(f"{key}={operand.code}" for key, operand in kwarg_operands.items())
    return (
        ", ".join(parts),
        inputs,
        tuple(operand.value for operand in arg_operands),
        {key: operand.value for key, operand in kwarg_operands.items()},
        copyable,
        code_prelude,
    )


def _structured_operation_from_call(
    call: provenance.ConsoleCall,
) -> provenance.ToolProvenanceOperation | None:
    with contextlib.suppress(Exception):
        return provenance.operation_from_console_call(call)
    return None


def _structured_seed_and_operations(
    source: _ConsoleOperand,
    operation: provenance.ToolProvenanceOperation | None,
) -> tuple[
    str | None,
    tuple[provenance.ToolProvenanceOperation, ...],
]:
    if operation is None or (source.seed_expression is None and not source.copyable):
        return None, ()
    return source.seed_expression or source.code, (*source.operations, operation)


class _ConsoleAccessorProxy:
    def __init__(
        self,
        owner: _ConsoleDataHandleBase,
        accessor: typing.Any,
        path: tuple[str, ...],
        expression: str,
    ) -> None:
        self._owner = owner
        self._accessor = accessor
        self._path = path
        self._expression = expression
        self.__wrapped__ = accessor
        for attr in ("__doc__", "__module__", "__name__", "__qualname__"):
            with contextlib.suppress(Exception):
                value = getattr(accessor, attr)
                if value is not None:
                    setattr(self, attr, value)
        with contextlib.suppress(TypeError, ValueError):
            self.__signature__ = inspect.signature(accessor)

    def _call(
        self,
        func: Callable[..., typing.Any],
        expression: str,
        path: tuple[str, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        (
            call_code,
            call_inputs,
            raw_args,
            raw_kwargs,
            args_copyable,
            call_prelude,
        ) = _format_call_code(args, kwargs)
        source_operand = self._owner._console_operand()
        result = func(*raw_args, **raw_kwargs)
        inputs, copyable, code_prelude = _merge_operands(
            source_operand,
            _ConsoleOperand(None, "", call_inputs, args_copyable, call_prelude),
        )
        call = provenance.ConsoleCall(
            func=func,
            accessor_path=path,
            args=raw_args,
            kwargs=raw_kwargs,
            display_code=f"{expression}({call_code})",
            has_extra_tracked_inputs=bool(call_inputs),
            receiver_data=source_operand.value,
        )
        seed_expression, operations = _structured_seed_and_operations(
            source_operand, _structured_operation_from_call(call)
        )
        return self._owner._wrap_console_result(
            result,
            call.display_code,
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
            operations=operations,
            seed_expression=seed_expression,
        )

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if not callable(self._accessor):
            raise TypeError(f"{self._expression} is not callable")
        return self._call(
            self._accessor,
            self._expression,
            self._path,
            *args,
            **kwargs,
        )

    def __getattr__(self, attr: str) -> typing.Any:
        value = getattr(self._accessor, attr)
        expression = f"{self._expression}.{attr}"
        path = (*self._path, attr)
        if not callable(value):
            if isinstance(value, xr.DataArray):
                operand = self._owner._console_operand()
                return self._owner._wrap_console_result(
                    value,
                    expression,
                    operand.script_inputs,
                    copyable=operand.copyable,
                    code_prelude=operand.code_prelude,
                )
            return value

        @functools.wraps(value)
        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            return self._call(value, expression, path, *args, **kwargs)

        return _method

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._accessor)))


class _ConsoleCoarsenProxy:
    def __init__(
        self,
        owner: _ConsoleDataHandleBase,
        coarsened: typing.Any,
        source_operand: _ConsoleOperand,
        expression: str,
        raw_args: tuple[typing.Any, ...],
        raw_kwargs: dict[str, typing.Any],
        script_inputs: Sequence[provenance.ScriptInput],
        *,
        copyable: bool,
        code_prelude: Sequence[str],
    ) -> None:
        self._owner = owner
        self._coarsened = coarsened
        self._source_operand = source_operand
        self._expression = expression
        self._raw_args = raw_args
        self._raw_kwargs = raw_kwargs
        self._script_inputs = tuple(script_inputs)
        self._copyable = copyable
        self._code_prelude = tuple(code_prelude)

    def __getattr__(self, attr: str) -> typing.Any:
        value = getattr(self._coarsened, attr)
        if not callable(value):
            return value

        @functools.wraps(value)
        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            (
                call_code,
                call_inputs,
                raw_args,
                raw_kwargs,
                args_copyable,
                call_prelude,
            ) = _format_call_code(args, kwargs)
            result = value(*raw_args, **raw_kwargs)
            inputs, copyable, code_prelude = _merge_operands(
                _ConsoleOperand(
                    None,
                    "",
                    self._script_inputs,
                    self._copyable,
                    self._code_prelude,
                ),
                _ConsoleOperand(None, "", call_inputs, args_copyable, call_prelude),
            )
            expression = f"{self._expression}.{attr}({call_code})"
            operation = None
            if not raw_args and not raw_kwargs:
                source_names = {
                    script_input.name
                    for script_input in self._source_operand.script_inputs
                }
                call = provenance.ConsoleCall(
                    dataarray_method="coarsen",
                    args=self._raw_args,
                    kwargs={**self._raw_kwargs, "_reducer": attr},
                    display_code=expression,
                    has_extra_tracked_inputs=any(
                        script_input.name not in source_names for script_input in inputs
                    ),
                    receiver_data=self._source_operand.value,
                )
                operation = _structured_operation_from_call(call)
            seed_expression, operations = _structured_seed_and_operations(
                self._source_operand, operation
            )
            return self._owner._wrap_console_result(
                result,
                expression,
                inputs,
                copyable=copyable,
                code_prelude=code_prelude,
                operations=operations,
                seed_expression=seed_expression,
            )

        with contextlib.suppress(TypeError, ValueError):
            typing.cast("typing.Any", _method).__signature__ = inspect.signature(value)
        return _method

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._coarsened)))


class _ConsoleModuleProxy(types.ModuleType):
    def __init__(self, module: types.ModuleType, alias: str) -> None:
        super().__init__(module.__name__, module.__doc__)
        self._module = module
        self._alias = alias
        for attr in ("__file__", "__package__", "__spec__", "__all__"):
            if hasattr(module, attr):
                setattr(self, attr, getattr(module, attr))

    def __getattr__(self, attr: str) -> typing.Any:
        value = getattr(self._module, attr)
        expression = f"{self._alias}.{attr}"
        if isinstance(value, types.ModuleType):
            return _ConsoleModuleProxy(value, expression)
        if isinstance(value, type):
            return value
        if not callable(value):
            return value

        @functools.wraps(value)
        def _function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            (
                call_code,
                call_inputs,
                raw_args,
                raw_kwargs,
                args_copyable,
                call_prelude,
            ) = _format_call_code(args, kwargs)
            result = value(*raw_args, **raw_kwargs)
            if not call_inputs:
                return result
            source_operand = None
            call_args = raw_args
            call_kwargs = raw_kwargs
            with contextlib.suppress(TypeError, ValueError):
                signature = inspect.signature(value)
                bound_args = signature.bind_partial(*args, **kwargs).arguments
                raw_bound_args = signature.bind_partial(
                    *raw_args, **raw_kwargs
                ).arguments
                source_param = next(
                    (
                        name
                        for name, arg in bound_args.items()
                        if isinstance(arg, _ConsoleDataHandleBase)
                    ),
                    None,
                )
                if source_param is not None:
                    source = bound_args[source_param]
                    if isinstance(source, _ConsoleDataHandleBase):
                        source_operand = source._console_operand()
                        call_args = ()
                        call_kwargs = {
                            key: arg
                            for key, arg in raw_bound_args.items()
                            if key != source_param
                        }
            operation = None
            seed_expression = None
            operations: tuple[
                provenance.ToolProvenanceOperation,
                ...,
            ] = ()
            if source_operand is not None:
                source_inputs = source_operand.script_inputs
                source_names = {script_input.name for script_input in source_inputs}
                call = provenance.ConsoleCall(
                    func=value,
                    args=call_args,
                    kwargs=call_kwargs,
                    display_code=f"{expression}({call_code})",
                    has_extra_tracked_inputs=any(
                        script_input.name not in source_names
                        for script_input in call_inputs
                    ),
                    receiver_data=source_operand.value,
                )
                operation = _structured_operation_from_call(call)
                seed_expression, operations = _structured_seed_and_operations(
                    source_operand, operation
                )
            owner = _first_console_handle(args)
            if owner is None:
                owner = _first_console_handle(kwargs)
            if owner is None:
                return result
            return owner._wrap_console_result(
                result,
                f"{expression}({call_code})",
                call_inputs,
                copyable=args_copyable,
                code_prelude=call_prelude,
                operations=operations,
                seed_expression=seed_expression,
            )

        return _function

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._module)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ConsoleModuleProxy):
            other = other._module
        return self._module == other

    def __hash__(self) -> int:
        return hash(self._module)


class _ConsoleFunctionProxy:
    _erlab_console_function_proxy = True

    def __init__(self, function: typing.Any) -> None:
        self._function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        (
            call_code,
            call_inputs,
            raw_args,
            raw_kwargs,
            args_copyable,
            call_prelude,
        ) = _format_call_code(args, kwargs)
        result = self._function(*raw_args, **raw_kwargs)
        owner = _first_console_handle(args)
        if owner is None:
            owner = _first_console_handle(kwargs)
        if owner is None:
            return result

        function_operand = _callable_operand(self)
        function_code = typing.cast("str", getattr(self, "__name__", ""))
        if not function_code:
            function_code = typing.cast("str", self._function.__name__)
        function_copyable = False
        function_prelude: tuple[str, ...] = ()
        if function_operand is not None:
            function_code = function_operand.code
            function_copyable = function_operand.copyable
            function_prelude = function_operand.code_prelude

        return owner._wrap_console_result(
            result,
            f"{function_code}({call_code})",
            call_inputs,
            copyable=function_copyable and args_copyable,
            code_prelude=_dedupe_code_preludes(function_prelude, call_prelude),
        )


class _ConsoleDataHandleBase:
    __array_priority__ = 1000

    @property
    def data(self) -> xr.DataArray:
        raise NotImplementedError

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        raise NotImplementedError

    @property
    def _console_is_source(self) -> bool:
        return False

    def _console_operand(self) -> _ConsoleOperand:
        raise NotImplementedError

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> provenance.ToolProvenanceSpec | None:
        raise NotImplementedError

    def _wrap_console_result(
        self,
        value: typing.Any,
        expression: str,
        script_inputs: Sequence[provenance.ScriptInput],
        *,
        copyable: bool,
        code_prelude: Sequence[str] = (),
        operations: Sequence[provenance.ToolProvenanceOperation] = (),
        seed_expression: str | None = None,
    ) -> typing.Any:
        if not isinstance(value, xr.DataArray):
            return value
        return _DerivedDataNamespace(
            self._console_tools,
            value,
            expression,
            _dedupe_script_inputs(script_inputs),
            copyable=copyable,
            code_prelude=_dedupe_code_preludes(code_prelude),
            operations=operations,
            seed_expression=seed_expression,
        )

    def _binary_operation(
        self,
        other: typing.Any,
        symbol: str,
        operation: Callable[[typing.Any, typing.Any], typing.Any],
        *,
        reflected: bool = False,
    ) -> typing.Any:
        left_operand = (
            _operand_from_value(other) if reflected else self._console_operand()
        )
        right_operand = (
            self._console_operand() if reflected else _operand_from_value(other)
        )
        inputs, copyable, code_prelude = _merge_operands(left_operand, right_operand)
        result = operation(left_operand.value, right_operand.value)
        expression = f"{left_operand.code} {symbol} {right_operand.code}"
        return self._wrap_console_result(
            result,
            expression,
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
        )

    def _unary_operation(
        self, symbol: str, operation: Callable[[typing.Any], typing.Any]
    ) -> typing.Any:
        operand = self._console_operand()
        result = operation(operand.value)
        return self._wrap_console_result(
            result,
            f"{symbol}{operand.code}",
            operand.script_inputs,
            copyable=operand.copyable,
            code_prelude=operand.code_prelude,
        )

    def __add__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "+", operator.add)

    def __radd__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "+", operator.add, reflected=True)

    def __sub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "-", operator.sub)

    def __rsub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "-", operator.sub, reflected=True)

    def __mul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "*", operator.mul)

    def __rmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "*", operator.mul, reflected=True)

    def __matmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "@", operator.matmul)

    def __rmatmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "@", operator.matmul, reflected=True)

    def __truediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "/", operator.truediv)

    def __rtruediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "/", operator.truediv, reflected=True)

    def __floordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "//", operator.floordiv)

    def __rfloordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "//", operator.floordiv, reflected=True)

    def __mod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "%", operator.mod)

    def __rmod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "%", operator.mod, reflected=True)

    def __pow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "**", operator.pow)

    def __rpow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "**", operator.pow, reflected=True)

    def __and__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "&", operator.and_)

    def __rand__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "&", operator.and_, reflected=True)

    def __or__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "|", operator.or_)

    def __ror__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "|", operator.or_, reflected=True)

    def __xor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "^", operator.xor)

    def __rxor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "^", operator.xor, reflected=True)

    def __lt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "<", operator.lt)

    def __le__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, "<=", operator.le)

    def __gt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, ">", operator.gt)

    def __ge__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, ">=", operator.ge)

    def __eq__(self, other: object) -> typing.Any:
        return self._binary_operation(other, "==", operator.eq)

    def __ne__(self, other: object) -> typing.Any:
        return self._binary_operation(other, "!=", operator.ne)

    def __neg__(self) -> typing.Any:
        return self._unary_operation("-", operator.neg)

    def __pos__(self) -> typing.Any:
        return self._unary_operation("+", operator.pos)

    def __abs__(self) -> typing.Any:
        operand = self._console_operand()
        return self._wrap_console_result(
            abs(operand.value),
            f"abs({operand.code})",
            operand.script_inputs,
            copyable=operand.copyable,
            code_prelude=operand.code_prelude,
        )

    def __invert__(self) -> typing.Any:
        return self._unary_operation("~", operator.invert)

    def __array__(
        self,
        dtype: np.dtype[typing.Any] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        if copy is None:
            return np.asarray(self.data, dtype=dtype)
        return np.array(self.data, dtype=dtype, copy=copy)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if method != "__call__":
            return NotImplemented
        input_operands = tuple(_operand_from_value(value) for value in inputs)
        kwarg_operands = {
            key: _operand_from_value(value) for key, value in kwargs.items()
        }
        script_inputs, copyable, code_prelude = _merge_operands(
            *input_operands, *tuple(kwarg_operands.values())
        )
        raw_kwargs = {key: operand.value for key, operand in kwarg_operands.items()}
        result = ufunc(*(operand.value for operand in input_operands), **raw_kwargs)
        arg_code = ", ".join(
            (
                *(operand.code for operand in input_operands),
                *(f"{key}={operand.code}" for key, operand in kwarg_operands.items()),
            )
        )
        return self._wrap_console_result(
            result,
            f"np.{ufunc.__name__}({arg_code})",
            script_inputs,
            copyable=copyable,
            code_prelude=code_prelude,
        )

    def __getitem__(self, key: typing.Any) -> typing.Any:
        key_operand = _operand_from_value(key)
        self_operand = self._console_operand()
        result = self.data[key_operand.value]
        inputs, copyable, code_prelude = _merge_operands(self_operand, key_operand)
        return self._wrap_console_result(
            result,
            f"{self_operand.code}[{key_operand.code}]",
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
        )

    def qshow(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        tools = self._console_tools
        if tools is None:
            return self.data.qshow(*args, **kwargs)
        return tools._qshow_handle(self, *args, **kwargs)

    def __getattr__(self, attr: str) -> typing.Any:
        data_attr = getattr(self.data, attr)
        if attr in {"qsel", "qshow", "qplot", "qinfo", "kspace", "xlm", "modelfit"}:
            operand = self._console_operand()
            return _ConsoleAccessorProxy(
                self,
                data_attr,
                (attr,),
                f"{operand.code}.{attr}",
            )
        if not callable(data_attr):
            if isinstance(data_attr, xr.DataArray):
                operand = self._console_operand()
                return self._wrap_console_result(
                    data_attr,
                    f"{operand.code}.{attr}",
                    operand.script_inputs,
                    copyable=operand.copyable,
                    code_prelude=operand.code_prelude,
                )
            return data_attr

        @functools.wraps(data_attr)
        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            (
                call_code,
                call_inputs,
                raw_args,
                raw_kwargs,
                args_copyable,
                call_prelude,
            ) = _format_call_code(args, kwargs)
            self_operand = self._console_operand()
            result = data_attr(*raw_args, **raw_kwargs)
            inputs, copyable, code_prelude = _merge_operands(
                self_operand,
                _ConsoleOperand(None, "", call_inputs, args_copyable, call_prelude),
            )
            expression = f"{self_operand.code}.{attr}({call_code})"
            if attr == "coarsen" and not isinstance(result, xr.DataArray):
                return _ConsoleCoarsenProxy(
                    self,
                    result,
                    self_operand,
                    expression,
                    raw_args,
                    raw_kwargs,
                    inputs,
                    copyable=copyable,
                    code_prelude=code_prelude,
                )
            call = provenance.ConsoleCall(
                func=data_attr,
                dataarray_method=attr,
                args=raw_args,
                kwargs=raw_kwargs,
                display_code=expression,
                has_extra_tracked_inputs=bool(call_inputs),
                receiver_data=self_operand.value,
            )
            seed_expression, operations = _structured_seed_and_operations(
                self_operand, _structured_operation_from_call(call)
            )
            return self._wrap_console_result(
                result,
                expression,
                inputs,
                copyable=copyable,
                code_prelude=code_prelude,
                operations=operations,
                seed_expression=seed_expression,
            )

        with contextlib.suppress(TypeError, ValueError):
            typing.cast("typing.Any", _method).__signature__ = inspect.signature(
                data_attr
            )
        return _method

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        with contextlib.suppress(Exception):
            names.update(dir(self.data))
        return sorted(names)


class ToolNamespace(_ConsoleDataHandleBase):
    """Provenance-aware console handle for one managed ImageTool.

    In the manager console, ``tools[idx]`` accesses a top-level ImageTool, and
    ``tools[idx].children[j]`` accesses a child ImageTool in manager tree order. The
    handle delegates DataArray attributes, methods, operators, and supported ERLab
    calls to the underlying array while preserving provenance for derived results.
    ``.data`` remains exact current :class:`xarray.DataArray` access for compatibility
    and is not provenance-aware.

    Examples
    --------
    - Build a provenance-backed result:

      >>> tools[0] - tools[1]

    - Access the current DataArray of an ImageTool:

      >>> tools[1].data

    - Setting a new DataArray:

      >>> tools[1].data = new_data

    """

    def __init__(
        self,
        wrapper: _ImageToolWrapper | _ManagedWindowNode,
        tools: ToolsNamespace | None = None,
    ) -> None:
        self._wrapper_ref = weakref.ref(wrapper)
        self._tools_ref = weakref.ref(tools) if tools is not None else None

    @property
    def _wrapper(self) -> _ImageToolWrapper | _ManagedWindowNode:
        wrapper = self._wrapper_ref()
        if wrapper:
            return wrapper
        raise LookupError("Parent was destroyed")

    @property
    def tool(self) -> ImageTool:
        """The underlying ImageTool object."""
        return typing.cast("ImageTool", self._wrapper.imagetool)

    @property
    def data(self) -> xr.DataArray:
        """Current :class:`xarray.DataArray` displayed by the ImageTool."""
        return self.tool.slicer_area.displayed_data

    @data.setter
    def data(self, value: xr.DataArray | _ConsoleDataHandleBase) -> None:
        provenance_spec = None
        if isinstance(value, _ConsoleDataHandleBase):
            raw_value = value.data
            provenance_spec = value._console_provenance_spec(
                active_name=self._console_input_name,
                label=f"Replace {self._console_label} data from console",
            )
        else:
            raw_value = value
        if provenance_spec is not None:
            self._wrapper.set_detached_provenance(provenance_spec)
        self.tool.slicer_area.replace_source_data(raw_value, emit_edited=True)

    @property
    def children(self) -> _ToolChildren:
        """Child ImageTools displayed under this ImageTool in manager tree order."""
        return _ToolChildren(self)

    def _get_data_item(self, key):
        """Return a subset of the tool data."""
        return self.tool.slicer_area.displayed_data[key]

    def _prepare_data_item_assignment(self, key, value) -> xr.DataArray | None:
        """Validate a console item assignment before changing provenance."""
        value = _unwrap_console_value(value)
        slicer_area = self.tool.slicer_area
        if not slicer_area.has_active_filter:
            return None
        data = slicer_area.displayed_data.copy(deep=True)
        data[key] = value
        return data

    def _set_data_item(
        self,
        key,
        value,
        *,
        prepared_data: xr.DataArray | None = None,
        emit_signals: bool = True,
    ) -> None:
        """Safely mutate a subset of the tool data from the console."""
        value = _unwrap_console_value(value)
        slicer_area = self.tool.slicer_area
        if prepared_data is not None:
            slicer_area.replace_source_data(prepared_data, emit_edited=emit_signals)
            return
        slicer_area._set_source_item(key, value, emit_signals=emit_signals)

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        if self._tools_ref is None:
            return None
        return self._tools_ref()

    @property
    def _console_is_source(self) -> bool:
        return True

    @property
    def _console_input_name(self) -> str:
        tools = self._console_tools
        if tools is not None:
            return tools._node_data_name(self._wrapper)
        return _tool_data_name(typing.cast("_ImageToolWrapper", self._wrapper).index)

    @property
    def _console_label(self) -> str:
        tools = self._console_tools
        if tools is not None:
            return tools._node_label(self._wrapper, include_name=False)
        return f"ImageTool {typing.cast('_ImageToolWrapper', self._wrapper).index}"

    def _script_input(
        self,
    ) -> provenance.ScriptInput:
        label = self._console_label
        if self._wrapper.name:
            label += f": {self._wrapper.name}"
        wrapper_provenance = self._wrapper.displayed_provenance_spec
        provenance_spec = (
            wrapper_provenance.model_dump(mode="json")
            if wrapper_provenance is not None
            else None
        )
        return provenance.ScriptInput(
            name=self._console_input_name,
            label=label,
            node_uid=self._wrapper.uid,
            node_snapshot_token=self._wrapper.snapshot_token,
            provenance_spec=provenance_spec,
        )

    def _console_operand(self) -> _ConsoleOperand:
        return _ConsoleOperand(
            self.data,
            self._console_input_name,
            (self._script_input(),),
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> provenance.ToolProvenanceSpec | None:
        return self._wrapper.displayed_provenance_spec

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        key_operand = _operand_from_value(key)
        value_operand = _operand_from_value(value)
        self_operand = self._console_operand()
        script_inputs, copyable, code_prelude = _merge_operands(
            self_operand, key_operand, value_operand
        )
        target_name = self._console_input_name
        code = _script_code(
            code_prelude,
            "\n".join(
                (
                    f"{target_name} = {self_operand.code}.copy(deep=True)",
                    f"{target_name}[{key_operand.code}] = {value_operand.code}",
                )
            ),
        )
        provenance_spec = provenance.script(
            provenance.ScriptCodeOperation(
                label=f"Set {self._console_label} data item from console",
                code=code,
                copyable=copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=target_name,
            script_inputs=script_inputs,
        )
        prepared_data = self._prepare_data_item_assignment(
            key_operand.value, value_operand.value
        )
        if prepared_data is None:
            self._set_data_item(
                key_operand.value,
                value_operand.value,
                emit_signals=False,
            )
            self._wrapper.set_detached_provenance(provenance_spec)
            slicer_area = self.tool.slicer_area
            slicer_area.sigSourceDataReplaced.emit(
                slicer_area._tool_source_parent_data()
            )
            slicer_area.sigDataEdited.emit()
            return
        self._wrapper.set_detached_provenance(provenance_spec)
        self._set_data_item(
            key_operand.value,
            value_operand.value,
            prepared_data=prepared_data,
        )

    def __getattr__(self, attr):  # implicitly wrap methods from ImageToolWrapper
        if hasattr(self._wrapper, attr):
            m = getattr(self._wrapper, attr)
            if callable(m):
                return m
        return super().__getattr__(attr)

    def __repr__(self) -> str:
        label = self._console_label
        if self._wrapper.name:
            label += f": {self._wrapper.name}"
        out = f"{label}\n"
        out += f"  Added: {self._wrapper.added_time_display}\n"
        out += f"  Linked: {self.tool.slicer_area.is_linked}\n"
        return out


class _ToolChildren:
    def __init__(self, parent: ToolNamespace) -> None:
        self._parent = parent

    def _nodes(self) -> list[_ManagedWindowNode]:
        tools = self._parent._console_tools
        if tools is None:
            return []
        return tools._child_imagetool_nodes(self._parent._wrapper)

    def __getitem__(self, index: int | slice) -> ToolNamespace | list[ToolNamespace]:
        tools = self._parent._console_tools
        if tools is None:
            raise LookupError("Parent was destroyed")
        nodes = self._nodes()
        if isinstance(index, slice):
            return [ToolNamespace(node, tools) for node in nodes[index]]
        return ToolNamespace(nodes[index], tools)

    def __iter__(self):
        tools = self._parent._console_tools
        if tools is None:
            return iter(())
        return (ToolNamespace(node, tools) for node in self._nodes())

    def __len__(self) -> int:
        return len(self._nodes())

    def __repr__(self) -> str:
        tools = self._parent._console_tools
        if tools is None:
            return "No child ImageTools"
        nodes = self._nodes()
        if not nodes:
            return f"No child ImageTools for {self._parent._console_label}"
        lines = [f"Child ImageTools for {self._parent._console_label}:"]
        for index, node in enumerate(nodes):
            lines.append(f"[{index}] {tools._node_label(node, include_name=True)}")
        return "\n".join(lines)


class _DerivedDataNamespace(_ConsoleDataHandleBase):
    def __init__(
        self,
        tools: ToolsNamespace | None,
        data: xr.DataArray,
        expression: str,
        script_inputs: Sequence[provenance.ScriptInput],
        *,
        copyable: bool,
        code_prelude: Sequence[str] = (),
        operations: Sequence[provenance.ToolProvenanceOperation] = (),
        seed_expression: str | None = None,
    ) -> None:
        self._tools_ref = weakref.ref(tools) if tools is not None else None
        self._data = data
        self._expression = expression
        self._script_inputs = _dedupe_script_inputs(script_inputs)
        self._copyable = copyable
        self._code_prelude = _dedupe_code_preludes(code_prelude)
        self._operations = tuple(operations)
        self._seed_expression = seed_expression
        self._console_name: str | None = None

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        if self._tools_ref is None:
            return None
        return self._tools_ref()

    def _set_console_name(self, name: str) -> None:
        if _replay_name_reserved(name):
            return
        self._console_name = name

    def _console_operand(self) -> _ConsoleOperand:
        if self._console_name is None:
            return _ConsoleOperand(
                self.data,
                _derived_operand_code(self._expression),
                self._script_inputs,
                self._copyable,
                self._code_prelude,
                self._seed_expression,
                self._operations,
            )
        provenance_spec = self._console_provenance_spec(
            active_name=self._console_name,
            label=f"Assign console variable {self._console_name!r}",
        )
        provenance_payload = (
            provenance_spec.model_dump(mode="json")
            if provenance_spec is not None
            else None
        )
        return _ConsoleOperand(
            self.data,
            self._console_name,
            (
                provenance.ScriptInput(
                    name=self._console_name,
                    label=f"console variable {self._console_name!r}",
                    provenance_spec=provenance_payload,
                ),
            ),
            provenance_spec is not None,
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> provenance.ToolProvenanceSpec | None:
        if not self._script_inputs:
            return None
        if self._operations and self._seed_expression is not None:
            seed_code = _script_code(
                self._code_prelude,
                f"{active_name} = {self._seed_expression}",
            )
            return provenance.script(
                *self._operations,
                start_label="Run ImageTool manager console code",
                seed_code=seed_code,
                active_name=active_name,
                script_inputs=self._script_inputs,
            )
        code = _script_code(self._code_prelude, f"{active_name} = {self._expression}")
        return provenance.script(
            provenance.ScriptCodeOperation(
                label=label,
                code=code,
                copyable=self._copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=active_name,
            script_inputs=self._script_inputs,
        )

    def __repr__(self) -> str:
        return repr(self.data)


class ToolsNamespace:
    """A console interface that represents the ImageToolManager and its tools.

    In the manager console, this namespace can be accessed with the variable ``tools``.
    Integer indexing returns provenance-aware top-level ImageTool handles; selected
    child ImageTools are available through :attr:`selected`, and raw arrays through
    :attr:`selected_data`.

    Examples
    --------
    - Print the list of tools:

      >>> tools

    - Access :class:`ToolNamespace` by index:

      >>> tools[1]

    """

    def __init__(self, manager: ImageToolManager) -> None:
        self._manager_ref = weakref.ref(manager)
        self._shell_ref: weakref.ReferenceType[InteractiveShell] | None = None
        self._namespace_snapshot: dict[str, int] = {}

    @property
    def _manager(self) -> ImageToolManager:
        """Access the ImageToolManager instance."""
        manager = self._manager_ref()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

    @property
    def selected_data(self) -> list[xr.DataArray]:
        """Current DataArrays from selected root or child ImageTools."""
        return [
            self._manager.get_imagetool(target).slicer_area.displayed_data
            for target in self._manager._selected_imagetool_targets()
        ]

    @property
    def selected(self) -> list[ToolNamespace]:
        """Provenance-aware handles for selected root or child ImageTools."""
        return [
            ToolNamespace(self._manager._node_for_target(target), self)
            for target in self._manager._selected_imagetool_targets()
        ]

    def __getitem__(self, index: int) -> ToolNamespace | None:
        """Access a top-level ImageTool by manager index."""
        if index not in self._manager._tool_graph.root_wrappers:
            print(f"Tool {index} not found")
            return None

        return ToolNamespace(self._manager._tool_graph.root_wrappers[index], self)

    def _node_path(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[int] | None:
        path: list[int] = []
        current = node
        while current.parent_uid is not None:
            image_parent = self._image_parent_node(current)
            if image_parent is None:
                return None
            child_nodes = self._child_imagetool_nodes(image_parent)
            child_uids = [child.uid for child in child_nodes]
            if current.uid not in child_uids:
                return None
            child_index = child_uids.index(current.uid)
            path.append(child_index)
            current = image_parent
        for root_index, wrapper in self._manager._tool_graph.root_wrappers.items():
            if wrapper.uid == current.uid:
                return [root_index, *reversed(path)]
        return None

    def _child_imagetool_nodes(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[_ManagedWindowNode]:
        output: list[_ManagedWindowNode] = []

        def collect(parent: _ImageToolWrapper | _ManagedWindowNode) -> None:
            for child_uid in parent._childtool_indices:
                child = self._manager._tool_graph.nodes.get(child_uid)
                if child is None:
                    continue
                if child.is_imagetool:
                    output.append(child)
                    continue
                collect(child)

        collect(node)
        return output

    def _image_parent_node(
        self, node: _ManagedWindowNode
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        parent_uid = node.parent_uid
        while parent_uid is not None:
            parent = self._manager._tool_graph.nodes.get(parent_uid)
            if parent is None:
                return None
            if parent.is_imagetool:
                return parent
            parent_uid = parent.parent_uid
        return None

    def _node_data_name(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        path = self._node_path(node)
        if path is not None:
            return "data_" + "_".join(str(index) for index in path)
        uid_name = "".join(char if char.isalnum() else "_" for char in node.uid)
        return f"data_{uid_name[:8]}"

    def _node_label(
        self, node: _ImageToolWrapper | _ManagedWindowNode, *, include_name: bool
    ) -> str:
        path = self._node_path(node)
        display_index = ".".join(str(index) for index in path) if path else node.uid[:8]
        label = f"ImageTool {display_index}"
        if include_name and node.name:
            label += f": {node.name}"
        return label

    def bind_shell(self, shell: InteractiveShell) -> None:
        self._shell_ref = weakref.ref(shell)
        shell.events.register("pre_run_cell", self._pre_run_cell)
        shell.events.register("post_run_cell", self._post_run_cell)

    def unbind_shell(self) -> None:
        if self._shell_ref is None:
            return
        shell = self._shell_ref()
        if shell is not None:
            with contextlib.suppress(KeyError, ValueError):
                shell.events.unregister("pre_run_cell", self._pre_run_cell)
            with contextlib.suppress(KeyError, ValueError):
                shell.events.unregister("post_run_cell", self._post_run_cell)
        self._shell_ref = None

    def _pre_run_cell(self, info: ExecutionInfo | None = None) -> None:
        if self._shell_ref is None:
            return
        shell = self._shell_ref()
        if shell is None:
            return
        self._namespace_snapshot = {
            name: id(value) for name, value in shell.user_ns.items()
        }

    def _post_run_cell(self, result: ExecutionResult | None = None) -> None:
        if result is not None and (
            result.error_before_exec is not None or result.error_in_exec is not None
        ):
            return
        if self._shell_ref is not None and (shell := self._shell_ref()) is not None:
            for name, value in shell.user_ns.items():
                if name.startswith("_") or self._namespace_snapshot.get(name) == id(
                    value
                ):
                    continue
                if (
                    inspect.isfunction(value)
                    and value.__module__ == "__main__"
                    and value.__qualname__ == value.__name__ == name
                    and not inspect.iscoroutinefunction(value)
                ):
                    shell.user_ns[name] = _ConsoleFunctionProxy(value)
                    continue
                if not isinstance(value, _DerivedDataNamespace):
                    continue
                value._set_console_name(name)
        if result is None or not isinstance(result.result, _DerivedDataNamespace):
            return
        self._show_handle(
            result.result,
            active_name=result.result._console_name or "derived",
            label="Evaluate console expression",
        )

    def _manager_argument_targets_this_manager(self, value: typing.Any) -> bool:
        if value is False:
            return False
        if value is None or value is True:
            return True
        try:
            target = operator.index(value)
        except TypeError:
            return False
        return target == getattr(self._manager, "manager_index", None)

    def _show_dataarray_with_provenance(
        self,
        data: xr.DataArray,
        provenance_spec: provenance.ToolProvenanceSpec,
        **kwargs,
    ) -> bool:
        display_kwargs = dict(kwargs)
        display_kwargs.pop("execute", None)
        tool = erlab.interactive.itool(
            data, manager=False, execute=False, **display_kwargs
        )
        if not isinstance(tool, erlab.interactive.imagetool.ImageTool):
            return False
        self._manager.add_imagetool(
            tool,
            show=True,
            activate=True,
            provenance_spec=provenance_spec,
        )
        return True

    def _show_handle(
        self,
        handle: _ConsoleDataHandleBase,
        *,
        active_name: str,
        label: str,
        **kwargs: typing.Any,
    ) -> bool:
        provenance_spec = handle._console_provenance_spec(
            active_name=active_name,
            label=label,
        )
        if provenance_spec is None:
            return False
        return self._show_dataarray_with_provenance(
            handle.data, provenance_spec, **kwargs
        )

    def itool(
        self, data: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        manager = kwargs.get("manager")
        replace = kwargs.get("replace")
        if (
            isinstance(data, _ConsoleDataHandleBase)
            and not args
            and replace is None
            and self._manager_argument_targets_this_manager(manager)
        ):
            display_kwargs = dict(kwargs)
            display_kwargs.pop("manager", None)
            display_kwargs.pop("replace", None)
            if self._show_handle(
                data,
                active_name=(
                    data._console_name
                    if isinstance(data, _DerivedDataNamespace)
                    and data._console_name is not None
                    else "derived"
                ),
                label="Evaluate console expression",
                **display_kwargs,
            ):
                return None
        return erlab.interactive.itool(_unwrap_console_value(data), *args, **kwargs)

    def _qshow_handle(
        self,
        data: _ConsoleDataHandleBase,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        manager = kwargs.get("manager")
        if not args and self._manager_argument_targets_this_manager(manager):
            display_kwargs = dict(kwargs)
            display_kwargs.pop("manager", None)
            if self._show_handle(
                data,
                active_name=(
                    data._console_name
                    if isinstance(data, _DerivedDataNamespace)
                    and data._console_name is not None
                    else "derived"
                ),
                label="Evaluate console expression",
                **display_kwargs,
            ):
                return None
        return data.data.qshow(*args, **kwargs)

    def __repr__(self) -> str:
        output = []
        for index, wrapper in self._manager._tool_graph.root_wrappers.items():
            output.append(f"{index}: {wrapper.name}")
        if not output:
            return "No tools"
        return "\n".join(output)


class _JupyterConsoleWidget(qtconsole.inprocess.QtInProcessRichJupyterWidget):
    """A Jupyter console widget for ImageToolManager.

    This widget is derived from qtconsole with some modifications such as:

    - Support for dark mode

    - Custom banner text

    - Lazy kernel initialization, including lazily evaluated namespace injection

    - Automated storing of data from ImageTools with the ``%store`` magic command


    Parameters
    ----------
    parent
        The parent widget for the console.
    namespace
        A dictionary of objects to inject into the console namespace. The keys are the
        names of the objects in the console, and the values are the objects themselves.
        If the value is a string, it is imported as a module upon kernel initialization,
        improving startup time for the ImageToolManager application.

    """

    def __init__(
        self, parent=None, namespace: dict[str, typing.Any] | None = None
    ) -> None:
        super().__init__(parent)
        self.kernel_manager = qtconsole.inprocess.QtInProcessKernelManager()
        self._namespace = namespace
        self._kernel_banner_default: str = ""
        self._kernel_initializing = False
        self._erlab_loader_name: str | None = None
        self._erlab_data_dir: str | None = None
        self._erlab_io_hooks_registered = False
        self._tools_namespace: ToolsNamespace | None = None

    def _restore_erlab_io_state(self, *args, **kwargs) -> None:
        erlab.io.set_loader(self._erlab_loader_name)
        erlab.io.set_data_dir(self._erlab_data_dir)

    def _persist_erlab_io_state(self, *args, **kwargs) -> None:
        loader = erlab.io.loaders.current_loader
        self._erlab_loader_name = None if loader is None else loader.name

        data_dir = erlab.io.loaders.current_data_dir
        self._erlab_data_dir = None if data_dir is None else str(data_dir)

    def _register_erlab_io_hooks(self) -> None:
        if self._erlab_io_hooks_registered or not self.kernel_manager.kernel:
            return

        shell = self.kernel_manager.kernel.shell
        shell.events.register("pre_run_cell", self._restore_erlab_io_state)
        shell.events.register("post_run_cell", self._persist_erlab_io_state)
        self._erlab_io_hooks_registered = True
        self._persist_erlab_io_state()

    def initialize_kernel(self) -> None:
        if self.kernel_manager.kernel or self._kernel_initializing:
            return
        self._kernel_initializing = True
        try:
            self.kernel_manager.start_kernel()
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()

            super().execute(r"%load_ext storemagic", hidden=True, interactive=False)
            super().execute(
                r"%load_ext erlab.interactive", hidden=True, interactive=False
            )

            if self._namespace is not None:
                resolved_namespace = _resolve_console_namespace(self._namespace)
                self.kernel_manager.kernel.shell.push(resolved_namespace)
                tools_namespace = resolved_namespace.get("tools")
                if isinstance(tools_namespace, ToolsNamespace):
                    self._tools_namespace = tools_namespace
                    tools_namespace.bind_shell(self.kernel_manager.kernel.shell)
                super().execute(
                    r"xr.set_options(keep_attrs=True)",
                    hidden=True,
                    interactive=False,
                )
            self._register_erlab_io_hooks()
        finally:
            self._kernel_initializing = False

    def execute(
        self, source: str | None = None, hidden: bool = False, interactive: bool = False
    ) -> None:
        if not self.kernel_manager.kernel and not self._kernel_initializing:
            self.initialize_kernel()
        super().execute(source, hidden=hidden, interactive=interactive)

    def store_data_as(self, tool_index: int, name: str) -> None:
        """Store the data in an ImageTool with IPython to reuse in other scripts."""
        self.initialize_kernel()
        store_commands = (
            f"{name} = tools[{tool_index}].data",
            f"get_ipython().run_line_magic('store', '{name}')",
            f"del {name}",
        )
        self.execute("\n".join(store_commands), hidden=True)

    @QtCore.Slot()
    def shutdown_kernel(self) -> None:
        if self.kernel_manager.kernel:
            if self._tools_namespace is not None:
                self._tools_namespace.unbind_shell()
                self._tools_namespace = None
            if self._erlab_io_hooks_registered:
                shell = self.kernel_manager.kernel.shell
                with contextlib.suppress(KeyError, ValueError):
                    shell.events.unregister(
                        "pre_run_cell", self._restore_erlab_io_state
                    )
                with contextlib.suppress(KeyError, ValueError):
                    shell.events.unregister(
                        "post_run_cell", self._persist_erlab_io_state
                    )
                self._erlab_io_hooks_registered = False
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

    def _banner_default(self) -> str:
        banner = super()._banner_default()
        return banner.strip() + f" | ERLabPy {erlab.__version__}\n"

    @property
    def kernel_banner(self) -> str:
        def _command_ansi(title: str, command_list: list[str]):
            out = f"\033[1m* {title}\033[0m"
            for command in command_list:
                out += f"\n  {command}"
            return out

        info_str = (
            _command_ansi(
                "Access raw data",
                [
                    "tools[<index>].data",
                    "tools[<index>].children[0].data",
                    "tools.selected_data",
                ],
            )
            + "\n"
            + _command_ansi(
                "Track provenance",
                [
                    "tools[0] - tools[1]",
                    "tools[0].children[0] - tools[1]",
                    "tools.selected[0]",
                ],
            )
            + "\n"
            + _command_ansi("Change data", ["tools[<index>].data = <value>"])
            + "\n"
            + _command_ansi(
                "Control window visibility",
                ["tools[<index>].show(), .close(), .dispose()"],
            )
            + "\n"
        )

        return f"{self._kernel_banner_default}{info_str}"

    @kernel_banner.setter
    def kernel_banner(self, value: str) -> None:
        self._kernel_banner_default = value

    def _update_colors(self) -> None:
        """Detect dark mode and update the console colors accordingly."""
        if self.kernel_manager.kernel:
            colors = "linux" if erlab.interactive.colors.is_dark_mode() else "lightbg"
            self.set_default_style(colors)
            self._syntax_style_changed()
            self._style_sheet_changed()
            self._execute(
                f"""
from IPython.core.ultratb import VerboseTB
if getattr(VerboseTB, 'tb_highlight_style', None) is not None:
    VerboseTB.tb_highlight_style = '{self.syntax_style}'
elif getattr(VerboseTB, '_tb_highlight_style', None) is not None:
    VerboseTB._tb_highlight_style = '{self.syntax_style}'
else:
    get_ipython().run_line_magic('colors', '{colors}')
del VerboseTB
""",
                True,
            )  # Adapted from qtconsole.mainwindow.MainWindow.set_syntax_style

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 186)


class _ImageToolManagerJupyterConsole(QtWidgets.QDockWidget):
    """A dock widget containing the Jupyter console."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__("Console", manager, flags=QtCore.Qt.WindowType.Window)
        tools_namespace = ToolsNamespace(manager)

        self._console_widget = _JupyterConsoleWidget(
            parent=self,
            namespace={
                "np": np,
                "xr": xr,
                "erlab": erlab,
                "eri": erlab.interactive,
                "itool": tools_namespace.itool,
                "tools": tools_namespace,
                "era": "erlab.analysis",
                "eplt": "erlab.plotting",
                "plt": "matplotlib.pyplot",
            },
        )
        qapp = QtWidgets.QApplication.instance()

        if qapp:
            # Shutdown kernel when application quits
            qapp.aboutToQuit.connect(self._console_widget.shutdown_kernel)

        self.setWidget(self._console_widget)
        manager.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self)
        self.setFloating(False)
        self.hide()

        # Start kernel when console is shown
        self._console_widget.installEventFilter(self)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            hasattr(self, "_console_widget")
            and obj == self._console_widget
            and event is not None
            and event.type() == QtCore.QEvent.Type.Show
        ):
            self._console_widget.initialize_kernel()
            self._console_widget._update_colors()
        return super().eventFilter(obj, event)

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            self._console_widget._update_colors()

        super().changeEvent(evt)
