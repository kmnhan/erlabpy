"""Jupyter console widget for ImageToolManager."""

from __future__ import annotations

from erlab._provenance import Follow
from erlab.interactive.imagetool._provenance._code import (
    _expression_receiver_code,
    _replace_code_identifiers,
)
from erlab.interactive.imagetool._provenance._graph import _REPLAY_PROTECTED_NAMES
from erlab.interactive.imagetool._provenance._model import (
    ConsoleCall,
    ReplayStep,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _calculation_options,
    _SourceViewOperation,
    operation_from_console_call,
    recipe,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    ConcatOperation,
    ScriptCodeOperation,
    ShiftFromInputOperation,
)
from erlab.interactive.imagetool._provenance._program import (
    CalculatedInput,
    ConsoleInput,
    RecordedProgram,
)
from erlab.interactive.imagetool._provenance._trust import provenance_code_trust_entries
from erlab.interactive.imagetool.manager._data_graph import (
    _CalculatedResult,
    _InputRevision,
    _WindowInput,
)

__all__ = ["ToolNamespace", "ToolsNamespace", "_ImageToolManagerJupyterConsole"]

import atexit
import contextlib
import functools
import importlib
import inspect
import keyword
import operator
import os
import types
import typing
import unicodedata
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import qtconsole.inprocess
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.utils
from erlab._provenance._records import Recipe, capture_display_data
from erlab.interactive.imagetool import _kspace_conversion

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from IPython.core.interactiveshell import (
        ExecutionInfo,
        ExecutionResult,
        InteractiveShell,
    )

    from erlab.interactive._code_trust._core import CodeTrustEntry
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
        if _is_console_library_module(value):
            value = _ConsoleLibraryNamespace(value)
        resolved[name] = value
    return resolved


def _is_console_library_module(value: typing.Any) -> bool:
    return isinstance(value, types.ModuleType) and (
        value.__name__ in {"xarray", "erlab", "erlab.analysis"}
        or value.__name__.startswith("erlab.analysis.")
    )


class _ConsoleLibraryFunction:
    """Dispatch handle arguments without modifying the installed library function."""

    def __init__(self, func: Callable[..., typing.Any]) -> None:
        self._function = func
        functools.update_wrapper(self, func)

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        owner = _first_console_handle((args, kwargs))
        if owner is None:
            return self._function(*args, **kwargs)
        tools = owner._console_tools
        if tools is None:
            _snapshot_required("A library call whose Manager is unavailable")
        return tools.apply(self._function, *args, **kwargs)


class _ConsoleLibraryNamespace(types.ModuleType):
    """Provide direct recorded calls through the console's supplied library aliases."""

    def __init__(self, module: types.ModuleType) -> None:
        super().__init__(module.__name__, module.__doc__)
        self._module = module
        self._members: dict[str, tuple[typing.Any, typing.Any]] = {}

    def __getattr__(self, name: str) -> typing.Any:
        value = getattr(self._module, name)
        cached = self._members.get(name)
        if cached is not None and cached[0] is value:
            return cached[1]
        if _is_console_library_module(value):
            member: typing.Any = _ConsoleLibraryNamespace(value)
        elif value is xr.concat or (
            inspect.isfunction(value) and value.__module__.startswith("erlab.analysis.")
        ):
            member = _ConsoleLibraryFunction(value)
        else:
            return value
        self._members[name] = (value, member)
        return member

    def __dir__(self) -> list[str]:
        return sorted(dir(self._module))


def _tool_data_name(index: int) -> str:
    return f"data_{index}"


def _is_local_console_input(script_input: ScriptInput) -> bool:
    return script_input.node_uid is None and script_input.revision_id is None


def _console_authored_entries(
    manager: ImageToolManager,
    node: _ImageToolWrapper | _ManagedWindowNode,
    provenance_spec: ToolProvenanceSpec | None,
) -> tuple[CodeTrustEntry, ...]:
    return manager._workspace_controller._locate_imagetool_code_trust_entries(
        provenance_code_trust_entries(
            provenance_spec,
            location_prefix="runtime",
            include_inputs=_is_local_console_input,
        ),
        node_getter=lambda: node,
    )


def _dedupe_script_inputs(
    inputs: Sequence[ScriptInput],
) -> tuple[ScriptInput, ...]:
    deduped: list[ScriptInput] = []
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
    return (
        name in {"data", "derived", "tools"}
        or name in _REPLAY_PROTECTED_NAMES
        or name.startswith("data_")
        or not name.isidentifier()
        or keyword.iskeyword(name)
        or unicodedata.normalize("NFKC", name) != name
    )


def _snapshot_required(operation: str) -> typing.NoReturn:
    raise TypeError(
        f"{operation} is not supported on an ImageTool console handle. "
        "Use an independent snapshot (a = handle.data), then assign "
        "tools[index].data = a to apply edits to a window."
    )


@dataclass(frozen=True)
class _ConsoleOperand:
    value: typing.Any
    code: str
    script_inputs: tuple[ScriptInput, ...] = ()
    copyable: bool = True
    code_prelude: tuple[str, ...] = ()
    seed_expression: str | None = None
    operations: tuple[ToolProvenanceOperation, ...] = ()


def _expression_operand(operand: _ConsoleOperand) -> _ConsoleOperand:
    """Keep statement operations when a result becomes part of an expression."""
    if not any(operation.statement_mutates_input for operation in operand.operations):
        return operand
    name = "prepared_data"
    spec = script(
        *operand.operations,
        start_label="Prepare console input",
        seed_code=_script_code(
            operand.code_prelude, f"{name} = {operand.seed_expression}"
        ),
        active_name=name,
        script_inputs=operand.script_inputs,
    )
    item = ScriptInput(
        name=name,
        label="Prepared console input",
        provenance_spec=spec.model_dump(mode="json"),
    )
    retained = list(operand.script_inputs)
    weakref.finalize(item, retained.clear).atexit = False
    return _ConsoleOperand(operand.value, name, (item,), operand.copyable)


def _literal_code(value: typing.Any) -> tuple[str, bool]:
    if isinstance(value, np.ndarray | np.generic):
        if value.dtype.kind not in "biufcSU":
            return repr(value), False
        contents, copyable = _literal_code(value.tolist())
        code = f"np.array({contents}, dtype={value.dtype.str!r})"
        return (f"{code}[()]" if isinstance(value, np.generic) else code), copyable
    if isinstance(value, np.dtype) or (
        isinstance(value, type) and issubclass(value, np.generic)
    ):
        return f"np.dtype({np.dtype(value).str!r})", True
    if type(value) is float:
        if np.isnan(value):
            return "np.nan", True
        if np.isinf(value):
            return ("np.inf" if value > 0 else "-np.inf"), True
        return repr(value), True
    if type(value) is complex:
        real_code, _ = _literal_code(float(value.real))
        imag_code, _ = _literal_code(float(value.imag))
        return f"complex({real_code}, {imag_code})", True
    if value is None or type(value) in (bool, int, str, bytes):
        return repr(value), True
    if value is Ellipsis:
        return "...", True
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
    return "", False


def _merge_operands(
    *operands: _ConsoleOperand,
) -> tuple[
    tuple[_ConsoleOperand, ...],
    tuple[ScriptInput, ...],
    bool,
    tuple[str, ...],
]:
    """Bind each operand occurrence to its own name before composing the code."""
    reserved = {item.name for operand in operands for item in operand.script_inputs}
    inputs: list[ScriptInput] = []
    used: set[str] = set()
    resolved = []
    for operand in operands:
        replacements = {}
        operand_inputs = []
        for item in operand.script_inputs:
            name = item.name
            if not isinstance(item, ConsoleInput) and item in inputs:
                # A portable expression can repeat one external Python variable.
                operand_inputs.append(item)
                continue
            if name in used:
                suffix = 2
                while f"{name}_{suffix}" in reserved:
                    suffix += 1
                name = f"{name}_{suffix}"
            used.add(name)
            reserved.add(name)
            assigned = (
                item if name == item.name else item.model_copy(update={"name": name})
            )
            if assigned is not item and not isinstance(item, ConsoleInput):
                retained = [item]
                weakref.finalize(assigned, retained.clear).atexit = False
            inputs.append(assigned)
            operand_inputs.append(assigned)
            if assigned.name != item.name:
                replacements[item.name] = assigned.name
        if replacements:
            operand = replace(
                operand,
                code=_replace_code_identifiers(operand.code, replacements),
                script_inputs=tuple(operand_inputs),
                code_prelude=tuple(
                    _replace_code_identifiers(code, replacements)
                    for code in operand.code_prelude
                ),
                seed_expression=(
                    _replace_code_identifiers(operand.seed_expression, replacements)
                    if operand.seed_expression is not None
                    else None
                ),
                operations=tuple(
                    operation.model_copy(
                        update={
                            "code": _replace_code_identifiers(
                                operation.code, replacements
                            )
                        }
                    )
                    if isinstance(operation, ScriptCodeOperation)
                    and operation.code is not None
                    else operation
                    for operation in operand.operations
                ),
            )
        resolved.append(operand)
    return (
        tuple(resolved),
        tuple(inputs),
        all(operand.copyable for operand in resolved),
        _dedupe_code_preludes(*(operand.code_prelude for operand in resolved)),
    )


def _bound_library_operation(
    func: Callable[..., typing.Any], count: int, arguments: dict[str, typing.Any]
) -> ConcatOperation | ShiftFromInputOperation | None:
    """Use the same operations as the Manager dialogs for direct array inputs."""
    if func is xr.concat:
        options = {"dim", "coords", "compat", "join", "combine_attrs"}
        if arguments.keys() - options or not isinstance(arguments["dim"], str):
            return None
        # The public xarray signature resolves its combine defaults against
        # the user's active xarray options.
        defaults = inspect.signature(xr.concat).parameters
        settings = {
            name: arguments[name] if name in arguments else str(defaults[name].default)
            for name in options
        }
        try:
            return ConcatOperation.model_validate({"input_count": count, **settings})
        except ValueError:
            # Some xarray options (for example an explicit list of coordinates)
            # require the general recorded-call executor.
            return None
    options = dict(arguments)
    interpolation = options.pop("shift_kwargs", {})
    if interpolation.keys() - {"order", "mode", "cval", "prefilter"}:
        return None
    if interpolation.get("mode", "constant") != "constant":
        options["cval"] = 0.0
    return ShiftFromInputOperation(**(options | interpolation))


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
    if isinstance(value, _ConsoleOperand):
        return value
    if isinstance(value, _ConsoleDataHandleBase):
        return _expression_operand(value._console_operand())
    if isinstance(value, tuple):
        item_operands = tuple(_operand_from_value(item) for item in value)
        suffix = "," if len(item_operands) == 1 else ""
        item_operands, inputs, copyable, code_prelude = _merge_operands(*item_operands)
        return _ConsoleOperand(
            tuple(operand.value for operand in item_operands),
            f"({', '.join(operand.code for operand in item_operands)}{suffix})",
            inputs,
            copyable,
            code_prelude,
        )
    if isinstance(value, list):
        item_operands = tuple(_operand_from_value(item) for item in value)
        item_operands, inputs, copyable, code_prelude = _merge_operands(*item_operands)
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
        pair_operands, inputs, copyable, code_prelude = _merge_operands(*pair_operands)
        key_operands, value_operands = pair_operands[::2], pair_operands[1::2]
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
    if not copyable:
        _snapshot_required(f"An operand of type {type(value).__name__}")
    return _ConsoleOperand(value, code, copyable=copyable)


def _format_call_code(
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
    *,
    receiver: _ConsoleOperand | None = None,
) -> tuple[
    str,
    tuple[ScriptInput, ...],
    tuple[typing.Any, ...],
    dict[str, typing.Any],
    bool,
    tuple[str, ...],
]:
    arg_operands = tuple(_operand_from_value(arg) for arg in args)
    kwarg_operands = {key: _operand_from_value(value) for key, value in kwargs.items()}
    operands = (*arg_operands, *kwarg_operands.values())
    if receiver is not None:
        merged, _, _, _ = _merge_operands(receiver, *operands)
        operands = merged[1:]
    operands, inputs, copyable, code_prelude = _merge_operands(*operands)
    arg_operands = operands[: len(args)]
    kwarg_operands = dict(zip(kwargs, operands[len(args) :], strict=True))
    parts = [operand.code for operand in arg_operands]
    for key, operand in kwarg_operands.items():
        if (
            key.isidentifier()
            and not keyword.iskeyword(key)
            and unicodedata.normalize("NFKC", key) == key
        ):
            parts.append(f"{key}={operand.code}")
        else:
            parts.append(f"**{{{key!r}: {operand.code}}}")
    return (
        ", ".join(parts),
        inputs,
        tuple(operand.value for operand in arg_operands),
        {key: operand.value for key, operand in kwarg_operands.items()},
        copyable,
        code_prelude,
    )


def _structured_operation_from_call(
    call: ConsoleCall,
) -> ToolProvenanceOperation | None:
    with contextlib.suppress(Exception):
        return operation_from_console_call(call)
    return None


def _xarray_options(*operands: _ConsoleOperand) -> dict[str, typing.Any]:
    current = xr.get_options()
    options = {"keep_attrs": current["keep_attrs"]}
    if sum(isinstance(operand.value, xr.DataArray) for operand in operands) > 1:
        for key in ("arithmetic_join", "arithmetic_broadcast", "arithmetic_compat"):
            if key in current:
                options[key] = current[key]
    return options


def _apply_console_operation(
    operation: ToolProvenanceOperation, data: xr.DataArray
) -> xr.DataArray:
    return operation.apply(data if operation.reads_input_only else data.copy(deep=True))


def _structured_seed_and_operations(
    source: _ConsoleOperand,
    operation: ToolProvenanceOperation | None,
) -> tuple[
    str | None,
    tuple[ToolProvenanceOperation, ...],
]:
    if operation is None or (source.seed_expression is None and not source.copyable):
        return None, ()
    seed_expression = source.seed_expression or source.code
    return seed_expression, (*source.operations, operation)


def _console_callable(func: typing.Any, original: typing.Any) -> typing.Any:
    """Preserve call signatures without retaining a method bound to old data."""

    @functools.wraps(original, updated=())
    def wrapped(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return func(*args, **kwargs)

    wrapped.__wrapped__ = func
    with contextlib.suppress(TypeError, ValueError):
        signature = inspect.signature(original)
        func.__signature__ = signature
        typing.cast("typing.Any", wrapped).__signature__ = signature
    return wrapped


_CONSOLE_ACCESSOR_CALLS = frozenset(
    {
        ("qsel",),
        ("qsel", "average"),
        ("qsel", "mean"),
        ("qsel", "min"),
        ("qsel", "max"),
        ("qsel", "sum"),
        ("kspace", "as_configuration"),
        ("kspace", "set_normal"),
        ("kspace", "convert"),
    }
)

_CONSOLE_COARSEN_REDUCERS = frozenset(
    {"all", "any", "count", "max", "mean", "median", "min", "prod", "std", "sum", "var"}
)


class _ConsoleAccessorProxy:
    def __setattr__(self, attr: str, value: typing.Any) -> None:
        if not attr.startswith("_"):
            _snapshot_required(f"Setting an accessor attribute ({attr})")
        object.__setattr__(self, attr, value)

    def __init__(
        self,
        owner: _ConsoleDataHandleBase,
        path: tuple[str, ...],
    ) -> None:
        self._owner = owner
        self._path = path
        accessor = operator.attrgetter(".".join(path))(owner._console_operand().value)
        for attr in ("__doc__", "__module__", "__name__", "__qualname__"):
            with contextlib.suppress(Exception):
                value = getattr(accessor, attr)
                if value is not None:
                    setattr(self, attr, value)
        with contextlib.suppress(TypeError, ValueError):
            self.__signature__ = inspect.signature(accessor)
        self.__wrapped__ = _console_callable(
            functools.partial(self._call, owner, path), accessor
        )

    @staticmethod
    def _call(
        owner: _ConsoleDataHandleBase,
        path: tuple[str, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if path not in _CONSOLE_ACCESSOR_CALLS:
            _snapshot_required(".".join(path))
        source_operand = owner._console_operand()
        (
            call_code,
            call_inputs,
            raw_args,
            raw_kwargs,
            args_copyable,
            call_prelude,
        ) = _format_call_code(args, kwargs, receiver=source_operand)
        resolve = operator.attrgetter(".".join(path))
        func = resolve(source_operand.value)
        expression = f"{source_operand.code}.{'.'.join(path)}"
        if not callable(func):
            raise TypeError(f"{expression} is not callable")
        call = ConsoleCall(
            func=func,
            accessor_path=path,
            args=raw_args,
            kwargs=raw_kwargs,
            display_code=f"{expression}({call_code})",
            has_extra_tracked_inputs=bool(call_inputs),
            receiver_data=source_operand.value,
        )
        operation = _structured_operation_from_call(call)
        if operation is None and path[0] == "kspace":
            _snapshot_required(f"{'.'.join(path)} with these arguments")
        options = (
            {"keep_attrs": xr.get_options()["keep_attrs"]} if operation is None else {}
        )
        _, inputs, copyable, code_prelude = _merge_operands(
            source_operand,
            _ConsoleOperand(None, "", call_inputs, args_copyable, call_prelude),
        )
        seed_expression, operations = _structured_seed_and_operations(
            source_operand, operation
        )
        return owner._calculate_console_result(
            lambda: (
                func(*raw_args, **raw_kwargs)
                if operation is None
                else _apply_console_operation(operation, source_operand.value)
            ),
            call.display_code,
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
            operations=operations,
            seed_expression=seed_expression,
            xarray_options=options or None,
        )

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return self._call(self._owner, self._path, *args, **kwargs)

    def __getattr__(self, attr: str) -> typing.Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        path = (*self._path, attr)
        if path not in _CONSOLE_ACCESSOR_CALLS:
            _snapshot_required(".".join(path))
        operand = self._owner._console_operand()
        value = operator.attrgetter(".".join(path))(operand.value)

        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            return self._call(self._owner, path, *args, **kwargs)

        return _console_callable(_method, value)

    def __dir__(self) -> list[str]:
        return sorted(
            path[-1] for path in _CONSOLE_ACCESSOR_CALLS if path[:-1] == self._path
        )


class _ConsoleCoarsenProxy:
    def __setattr__(self, attr: str, value: typing.Any) -> None:
        if not attr.startswith("_"):
            _snapshot_required(f"Setting a coarsening attribute ({attr})")
        object.__setattr__(self, attr, value)

    def __init__(
        self,
        owner: _ConsoleDataHandleBase,
        coarsened: typing.Any,
        source_operand: _ConsoleOperand,
        expression: str,
        raw_args: tuple[typing.Any, ...],
        raw_kwargs: dict[str, typing.Any],
        script_inputs: Sequence[ScriptInput],
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
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr not in _CONSOLE_COARSEN_REDUCERS:
            _snapshot_required(f"coarsen.{attr}")
        value = getattr(self._coarsened, attr)

        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            (
                call_code,
                call_inputs,
                raw_args,
                raw_kwargs,
                args_copyable,
                call_prelude,
            ) = _format_call_code(
                args,
                kwargs,
                receiver=_ConsoleOperand(None, self._expression, self._script_inputs),
            )
            _, inputs, copyable, code_prelude = _merge_operands(
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
                call = ConsoleCall(
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
                if operation is not None:
                    operation = operation.model_copy(
                        update={
                            "keep_attrs": xr.get_options()["keep_attrs"] is not False
                        }
                    )
            options = (
                {"keep_attrs": xr.get_options()["keep_attrs"]}
                if operation is None
                else {}
            )
            seed_expression, operations = _structured_seed_and_operations(
                self._source_operand, operation
            )
            return self._owner._calculate_console_result(
                lambda: (
                    value(*raw_args, **raw_kwargs)
                    if operation is None
                    else _apply_console_operation(operation, self._source_operand.value)
                ),
                expression,
                inputs,
                copyable=copyable,
                code_prelude=code_prelude,
                operations=operations,
                seed_expression=seed_expression,
                xarray_options=options or None,
            )

        return _console_callable(_method, value)

    def __dir__(self) -> list[str]:
        return sorted(_CONSOLE_COARSEN_REDUCERS)


# These methods have a direct public expression and do not mutate the receiver.
# Other DataArray APIs remain available through the explicit snapshot boundary.
_CONSOLE_READ_ONLY_METHODS = frozenset(
    {
        "all",
        "any",
        "argmax",
        "argmin",
        "argsort",
        "assign_attrs",
        "assign_coords",
        "astype",
        "broadcast_like",
        "chunk",
        "clip",
        "coarsen",
        "compute",
        "conj",
        "conjugate",
        "copy",
        "count",
        "cumprod",
        "cumsum",
        "diff",
        "differentiate",
        "drop_duplicates",
        "drop_indexes",
        "drop_isel",
        "drop_sel",
        "drop_vars",
        "dropna",
        "expand_dims",
        "fillna",
        "integrate",
        "interp",
        "interp_like",
        "interpolate_na",
        "isin",
        "isnull",
        "isel",
        "item",
        "max",
        "mean",
        "median",
        "min",
        "notnull",
        "pad",
        "prod",
        "quantile",
        "rank",
        "reindex",
        "reindex_like",
        "rename",
        "roll",
        "round",
        "sel",
        "shift",
        "sortby",
        "squeeze",
        "stack",
        "std",
        "sum",
        "swap_dims",
        "transpose",
        "thin",
        "unstack",
        "var",
        "where",
    }
)

_CONSOLE_INSPECTION_ATTRIBUTES = frozenset(
    {
        "dims",
        "dtype",
        "ndim",
        "shape",
        "size",
        "nbytes",
        "chunks",
        "sizes",
        "chunksizes",
        "name",
    }
)


def _console_inspection(value: typing.Any) -> typing.Any:
    if value is None or isinstance(value, (str, bytes, int, float, complex, np.dtype)):
        return value
    if isinstance(value, tuple):
        return tuple(_console_inspection(item) for item in value)
    if isinstance(value, Mapping):
        return types.MappingProxyType(
            {
                _console_inspection(key): _console_inspection(item)
                for key, item in value.items()
            }
        )
    _snapshot_required(f"Inspection of {type(value).__name__}")


_OPERATOR_SYMBOLS: dict[Callable[..., typing.Any], str] = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.matmul: "@",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.eq: "==",
    operator.ne: "!=",
    operator.neg: "-",
    operator.pos: "+",
    operator.invert: "~",
}


class _ConsoleDataHandleBase:
    __array_priority__ = 1000

    def __setattr__(self, attr: str, value: typing.Any) -> None:
        if not attr.startswith("_") and not (
            attr == "data" and isinstance(self, ToolNamespace)
        ):
            _snapshot_required(f"Setting {attr}")
        object.__setattr__(self, attr, value)

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        _snapshot_required("Editing a calculated result")

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
    ) -> ToolProvenanceSpec | None:
        raise NotImplementedError

    def _calculate_console_result(
        self,
        detached: Callable[[], xr.DataArray],
        expression: str,
        script_inputs: Sequence[ScriptInput],
        *,
        copyable: bool,
        code_prelude: Sequence[str] = (),
        operations: Sequence[ToolProvenanceOperation] = (),
        seed_expression: str | None = None,
        xarray_options: dict[str, typing.Any] | None = None,
    ) -> typing.Any:
        return _DerivedDataNamespace(
            self._console_tools,
            detached,
            expression,
            _dedupe_script_inputs(script_inputs),
            copyable=copyable,
            code_prelude=_dedupe_code_preludes(code_prelude),
            operations=operations,
            seed_expression=seed_expression,
            xarray_options=xarray_options,
            previous=self,
        )

    def _binary_operation(
        self,
        other: typing.Any,
        operation: Callable[[typing.Any, typing.Any], typing.Any],
        *,
        reflected: bool = False,
    ) -> typing.Any:
        left_operand = (
            _operand_from_value(other) if reflected else _operand_from_value(self)
        )
        right_operand = (
            _operand_from_value(self) if reflected else _operand_from_value(other)
        )
        (left_operand, right_operand), inputs, copyable, code_prelude = _merge_operands(
            left_operand, right_operand
        )
        options = _xarray_options(left_operand, right_operand)
        symbol = _OPERATOR_SYMBOLS[operation]
        left_code = (
            _expression_receiver_code(left_operand.code)
            if symbol == "**"
            else left_operand.code
        )
        expression = f"{left_code} {symbol} {right_operand.code}"
        return self._calculate_console_result(
            lambda: operation(left_operand.value, right_operand.value),
            expression,
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
            xarray_options=options,
        )

    def _unary_operation(
        self, operation: Callable[[typing.Any], typing.Any]
    ) -> typing.Any:
        operand = _operand_from_value(self)
        options = _xarray_options(operand)
        return self._calculate_console_result(
            lambda: operation(operand.value),
            f"{_OPERATOR_SYMBOLS[operation]}{operand.code}",
            operand.script_inputs,
            copyable=operand.copyable,
            code_prelude=operand.code_prelude,
            xarray_options=options,
        )

    def __add__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.add)

    def __radd__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.add, reflected=True)

    def __sub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.sub)

    def __rsub__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.sub, reflected=True)

    def __mul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.mul)

    def __rmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.mul, reflected=True)

    def __matmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.matmul)

    def __rmatmul__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.matmul, reflected=True)

    def __truediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.truediv)

    def __rtruediv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.truediv, reflected=True)

    def __floordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.floordiv)

    def __rfloordiv__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.floordiv, reflected=True)

    def __mod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.mod)

    def __rmod__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.mod, reflected=True)

    def __pow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.pow)

    def __rpow__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.pow, reflected=True)

    def __and__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.and_)

    def __rand__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.and_, reflected=True)

    def __or__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.or_)

    def __ror__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.or_, reflected=True)

    def __xor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.xor)

    def __rxor__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.xor, reflected=True)

    def __lt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.lt)

    def __le__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.le)

    def __gt__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.gt)

    def __ge__(self, other: typing.Any) -> typing.Any:
        return self._binary_operation(other, operator.ge)

    def __eq__(self, other: object) -> typing.Any:
        return self._binary_operation(other, operator.eq)

    def __ne__(self, other: object) -> typing.Any:
        return self._binary_operation(other, operator.ne)

    def __neg__(self) -> typing.Any:
        return self._unary_operation(operator.neg)

    def __pos__(self) -> typing.Any:
        return self._unary_operation(operator.pos)

    def __abs__(self) -> typing.Any:
        operand = _operand_from_value(self)
        options = _xarray_options(operand)
        return self._calculate_console_result(
            lambda: abs(operand.value),
            f"abs({operand.code})",
            operand.script_inputs,
            copyable=operand.copyable,
            code_prelude=operand.code_prelude,
            xarray_options=options,
        )

    def __invert__(self) -> typing.Any:
        return self._unary_operation(operator.invert)

    def __len__(self) -> int:
        return len(self._console_operand().value)

    def __bool__(self) -> bool:
        return bool(self._console_operand().value)

    def __float__(self) -> float:
        return float(self._console_operand().value)

    def __int__(self) -> int:
        return int(self._console_operand().value)

    def __complex__(self) -> complex:
        return complex(self._console_operand().value)

    def __array__(
        self,
        dtype: np.dtype[typing.Any] | None = None,
        copy: bool | None = None,
    ) -> typing.NoReturn:
        _snapshot_required("Raw NumPy conversion")

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if (
            method != "__call__"
            or ufunc.nout != 1
            or getattr(np, ufunc.__name__, None) is not ufunc
            or kwargs.get("out") is not None
            or kwargs.get("where", True) is not True
        ):
            _snapshot_required(f"np.{ufunc.__name__}.{method} with these arguments")
        input_operands = tuple(_operand_from_value(value) for value in inputs)
        kwarg_operands = {
            key: _operand_from_value(value) for key, value in kwargs.items()
        }
        operands, script_inputs, copyable, code_prelude = _merge_operands(
            *input_operands, *tuple(kwarg_operands.values())
        )
        input_operands = operands[: len(inputs)]
        kwarg_operands = dict(zip(kwargs, operands[len(inputs) :], strict=True))
        raw_kwargs = {key: operand.value for key, operand in kwarg_operands.items()}
        options = _xarray_options(*input_operands)
        arg_code = ", ".join(
            (
                *(operand.code for operand in input_operands),
                *(f"{key}={operand.code}" for key, operand in kwarg_operands.items()),
            )
        )
        return self._calculate_console_result(
            lambda: ufunc(*(operand.value for operand in input_operands), **raw_kwargs),
            f"np.{ufunc.__name__}({arg_code})",
            script_inputs,
            copyable=copyable,
            code_prelude=code_prelude,
            xarray_options=options,
        )

    def __getitem__(self, key: typing.Any) -> typing.Any:
        key_operand = _operand_from_value(key)
        self_operand = _operand_from_value(self)
        (self_operand, key_operand), inputs, copyable, code_prelude = _merge_operands(
            self_operand, key_operand
        )
        return self._calculate_console_result(
            lambda: self_operand.value[key_operand.value],
            f"{self_operand.code}[{key_operand.code}]",
            inputs,
            copyable=copyable,
            code_prelude=code_prelude,
        )

    def qshow(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        tools = self._console_tools
        if tools is None:
            _snapshot_required("Displaying a handle whose Manager is unavailable")
        return tools._qshow_handle(self, *args, **kwargs)

    def __getattr__(self, attr: str) -> typing.Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr in {"coords", "attrs", "encoding", "values", "variable", "loc"}:
            _snapshot_required(f"Access to {attr}")
        operand = self._console_operand()
        if attr in {"qsel", "kspace"}:
            return _ConsoleAccessorProxy(self, (attr,))
        if attr in _CONSOLE_INSPECTION_ATTRIBUTES:
            return _console_inspection(getattr(operand.value, attr))
        if attr in operand.value.coords or attr in {"T", "real", "imag"}:
            operand = _expression_operand(operand)
            return self._calculate_console_result(
                lambda: getattr(operand.value, attr),
                f"{operand.code}.{attr}"
                if attr.isidentifier()
                and not keyword.iskeyword(attr)
                and unicodedata.normalize("NFKC", attr) == attr
                else f"{operand.code}[{str(attr)!r}]",
                operand.script_inputs,
                copyable=operand.copyable,
                code_prelude=operand.code_prelude,
            )
        if attr not in _CONSOLE_READ_ONLY_METHODS:
            _snapshot_required(attr)
        data_attr = getattr(operand.value, attr)

        def _method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            self_operand = self._console_operand()
            (
                call_code,
                call_inputs,
                raw_args,
                raw_kwargs,
                args_copyable,
                call_prelude,
            ) = _format_call_code(args, kwargs, receiver=self_operand)
            receiver = self_operand.value
            func = getattr(receiver, attr)
            expression = f"{self_operand.code}.{attr}({call_code})"
            call = ConsoleCall(
                func=func,
                dataarray_method=attr,
                args=raw_args,
                kwargs=raw_kwargs,
                display_code=expression,
                has_extra_tracked_inputs=bool(call_inputs),
                receiver_data=receiver,
            )
            operation = _structured_operation_from_call(call)
            options = (
                {"keep_attrs": xr.get_options()["keep_attrs"]}
                if operation is None
                else {}
            )
            if operation is None:
                self_operand = _expression_operand(self_operand)
            (self_operand, call_operand), inputs, copyable, code_prelude = (
                _merge_operands(
                    self_operand,
                    _ConsoleOperand(
                        None, call_code, call_inputs, args_copyable, call_prelude
                    ),
                )
            )
            expression = f"{self_operand.code}.{attr}({call_operand.code})"
            if attr == "item":
                scalar = func(*raw_args, **raw_kwargs)
                return _console_inspection(
                    scalar.item() if isinstance(scalar, np.generic) else scalar
                )
            if attr == "coarsen":
                return _ConsoleCoarsenProxy(
                    self,
                    func(*raw_args, **raw_kwargs),
                    self_operand,
                    expression,
                    raw_args,
                    raw_kwargs,
                    inputs,
                    copyable=copyable,
                    code_prelude=code_prelude,
                )
            seed_expression, operations = _structured_seed_and_operations(
                self_operand, operation
            )
            return self._calculate_console_result(
                lambda: (
                    func(*raw_args, **raw_kwargs)
                    if operation is None
                    else _apply_console_operation(operation, receiver)
                ),
                expression,
                inputs,
                copyable=copyable,
                code_prelude=code_prelude,
                operations=operations,
                seed_expression=seed_expression,
                xarray_options=options or None,
            )

        return _console_callable(_method, data_attr)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        names.update(_CONSOLE_READ_ONLY_METHODS)
        names.update(_CONSOLE_INSPECTION_ATTRIBUTES)
        names.update({"qsel", "kspace", "T", "real", "imag"})
        with contextlib.suppress(Exception):
            names.update(str(name) for name in self._console_operand().value.coords)
        return sorted(names)


class ToolNamespace(_ConsoleDataHandleBase):
    """Provenance-aware console handle for one managed ImageTool.

    In the manager console, ``tools[idx]`` accesses a top-level ImageTool, and
    ``tools[idx].children[j]`` accesses a child manager row in tree order. Child rows
    that are ImageTools are also ``ToolNamespace`` handles; child rows that are
    analysis tools are navigable manager-row handles. Handles record supported array
    operations. Use ``dir(handle)`` to inspect them and :meth:`ToolsNamespace.apply`
    for supported library transformations. ``.data`` returns an independent
    :class:`xarray.DataArray` snapshot for ordinary xarray and NumPy use. Its edits
    are not tracked until it is assigned back to a window.

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
        self._tools = tools if tools is not None else ToolsNamespace(wrapper.manager)

    @property
    def _wrapper(self) -> _ImageToolWrapper | _ManagedWindowNode:
        wrapper = self._wrapper_ref()
        if wrapper and wrapper.manager._tool_graph.nodes.get(wrapper.uid) is wrapper:
            return wrapper
        raise LookupError(
            "This ImageTool window is closed. Obtain a new handle from tools[index]."
        )

    @property
    def tool(self) -> ImageTool:
        """The underlying ImageTool object."""
        return self._wrapper.manager.get_imagetool(self.uid)

    @property
    def window(self) -> ImageTool:
        """The underlying ImageTool window."""
        return self.tool

    @property
    def uid(self) -> str:
        """Manager node UID for this ImageTool row."""
        return self._wrapper.uid

    @property
    def name(self) -> str:
        """Current manager row display name."""
        return self._wrapper.name

    @property
    def kind(self) -> str:
        """Manager row kind."""
        return "ImageTool"

    @property
    def is_imagetool(self) -> bool:
        """Whether this console row handle wraps an ImageTool."""
        return True

    @property
    def data(self) -> xr.DataArray:
        """Return an independent, editable snapshot of the displayed data.

        Assign a plain DataArray to this property to replace the window data with
        a new checkpoint. Use ``tools[index][key] = value`` to edit the window
        directly and record the operation.

        .. versionchanged:: 3.30.0
           Reading this property returns a snapshot. Item, coordinate, and attribute
           edits to the returned DataArray do not change the ImageTool.
        """
        return self._wrapper.manager._data_graph.console_data(self._wrapper)

    @data.setter
    def data(self, value: xr.DataArray | _ConsoleDataHandleBase) -> None:
        manager = self._wrapper.manager
        publication: _CalculatedResult | None = None
        source_owners: set[str] = set()
        if isinstance(value, _ConsoleDataHandleBase):
            operand = value._console_operand()
            if isinstance(value, ToolNamespace):
                if value._wrapper.manager is not manager:
                    _snapshot_required("Assignment from another Manager")
                publication = value._view_publication(
                    self._console_input_name, operand, replace_uid=self.uid
                )
                source_owners.add(value.uid)
            else:
                if (
                    value._console_tools is not None
                    and value._console_tools._manager is not manager
                ):
                    _snapshot_required("Assignment from another Manager")
                if (
                    isinstance(value, _DerivedDataNamespace)
                    and value._captured_input is not None
                ):
                    publication = _CalculatedResult(
                        value._captured_input._capture, self._console_input_name
                    )
                else:
                    _snapshot_required("Assignment of an unrecorded result")
        self._commit_console_data(
            publication
            if publication is not None
            else typing.cast("xr.DataArray", value),
            source_owners=source_owners,
            authored=isinstance(value, _DerivedDataNamespace),
        )

    def _commit_console_data(
        self,
        data: xr.DataArray | _CalculatedResult,
        *,
        source_owners: Iterable[str],
        authored: bool,
        item_edit: bool = False,
    ) -> None:
        if not isinstance(data, xr.DataArray | _CalculatedResult):
            raise TypeError("Assign a DataArray or a console handle to .data.")
        manager = self._wrapper.manager
        graph = manager._data_graph
        with manager._workspace_controller.local_provenance_change(
            edited_entries=lambda: (
                _console_authored_entries(
                    manager, self._wrapper, self._wrapper.provenance_spec
                )
                if authored
                else ()
            ),
            source_owners=lambda: {
                self.uid: graph.graph.follows(self.uid) | set(source_owners)
            },
        ):
            self._wrapper.publish_replacement(
                data,
                edit="item" if item_edit else "edited",
            )

    @property
    def children(self) -> _ToolChildren:
        """Child manager rows displayed under this ImageTool in manager tree order."""
        return _ToolChildren(self)

    def _get_data_item(self, key):
        """Return a subset of the tool data."""
        return self.tool.slicer_area.displayed_data[key]

    @property
    def _console_tools(self) -> ToolsNamespace:
        return self._tools

    @property
    def _console_is_source(self) -> bool:
        return True

    @property
    def _console_input_name(self) -> str:
        return self._tools._node_data_name(self._wrapper)

    @property
    def _console_label(self) -> str:
        return self._tools._node_label(self._wrapper, include_name=False)

    def _script_input(
        self,
    ) -> ScriptInput:
        graph = self._wrapper.manager._data_graph
        graph.require_published(self.uid)
        recorded = graph.owns_result(self.uid)
        item = self._wrapper.manager._lineage_controller._script_input_for_node(
            self._wrapper,
            name=self._console_input_name,
            input_type=ConsoleInput,
        )
        if not recorded:
            graph.capture_input(self._wrapper, item)
        return item

    def _view_publication(
        self, name: str, operand: _ConsoleOperand, *, replace_uid: str | None = None
    ) -> _CalculatedResult:
        item = operand.script_inputs[0]
        revision = typing.cast("ConsoleInput", item)._capture.revision_id
        adapter = self._wrapper.manager._data_graph
        source = _WindowInput(
            operand.value,
            _InputRevision(Follow(self.uid), revision),
            adapter.observe_input(self.uid),
        )
        return adapter.calculate(
            (_SourceViewOperation(source_kind="public_data"),),
            (source,),
            input_names=(item.name,),
            output_name=name,
            replace_uid=replace_uid,
            source_kind="public_data",
        )

    def _bound_data_recipe(
        self, active_name: str, source: ScriptInput
    ) -> ToolProvenanceSpec:
        return recipe(
            ReplayStep(operation=_SourceViewOperation(source_kind="public_data")),
            start_label=f"Use {self._console_label} data",
            active_name=active_name,
            script_inputs=(source,),
            primary_input=source.name,
        )

    def _console_operand(self) -> _ConsoleOperand:
        source = typing.cast("ConsoleInput", self._script_input())
        graph = self._wrapper.manager._data_graph.graph
        return _ConsoleOperand(
            graph.calculation_data(source._capture.revision_id),
            self._console_input_name,
            (source,),
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> ToolProvenanceSpec | None:
        return self._wrapper.displayed_provenance_spec

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        key_operand = _operand_from_value(key)
        value_operand = _operand_from_value(value)
        self_operand = self._console_operand()
        (
            (self_operand, key_operand, value_operand),
            script_inputs,
            copyable,
            code_prelude,
        ) = _merge_operands(self_operand, key_operand, value_operand)
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
        provenance_spec = script(
            ScriptCodeOperation(
                label=f"Set {self._console_label} data item from console",
                code=code,
                copyable=copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=target_name,
            script_inputs=script_inputs,
        )
        if not copyable:
            _snapshot_required("An item edit with an unrecorded operand")
        recorded = CalculatedInput(name=target_name)
        self._wrapper.manager._data_graph.calculate_console(
            RecordedProgram(spec=provenance_spec),
            script_inputs,
            recorded._capture,
        )
        self._commit_console_data(
            _CalculatedResult(recorded._capture, target_name),
            source_owners=(),
            authored=True,
            item_edit=True,
        )

    def __getattr__(self, attr):
        if attr in {"show", "hide", "dispose"}:
            return getattr(self._wrapper, attr)
        return super().__getattr__(attr)

    def __repr__(self) -> str:
        label = self._console_label
        if self._wrapper.name:
            label += f": {self._wrapper.name}"
        out = f"{label}\n"
        out += f"  Added: {self._wrapper.added_time_display}\n"
        out += f"  Linked: {self.tool.slicer_area.is_linked}\n"
        return out


class _ManagedToolNamespace:
    """Console handle for a non-ImageTool manager row."""

    def __init__(
        self,
        wrapper: _ManagedWindowNode,
        tools: ToolsNamespace | None = None,
    ) -> None:
        self._wrapper_ref = weakref.ref(wrapper)
        self._tools_ref = weakref.ref(tools) if tools is not None else None

    @property
    def _wrapper(self) -> _ManagedWindowNode:
        wrapper = self._wrapper_ref()
        if wrapper:
            return wrapper
        raise LookupError("Parent was destroyed")

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        if self._tools_ref is None:
            return None
        return self._tools_ref()

    @property
    def window(self) -> QtWidgets.QWidget | None:
        """The underlying managed tool window."""
        return self._wrapper.window

    @property
    def uid(self) -> str:
        """Manager node UID for this tool row."""
        return self._wrapper.uid

    @property
    def name(self) -> str:
        """Current manager row display name."""
        return self._wrapper.name

    @property
    def kind(self) -> str:
        """Manager row kind."""
        return self._wrapper.type_badge_text or "tool"

    @property
    def is_imagetool(self) -> bool:
        """Whether this console row handle wraps an ImageTool."""
        return False

    @property
    def children(self) -> _ToolChildren:
        """Child manager rows displayed under this tool row in manager tree order."""
        return _ToolChildren(self)

    @property
    def data(self) -> xr.DataArray:
        self._raise_not_imagetool()

    def _raise_not_imagetool(self) -> typing.NoReturn:
        tools = self._console_tools
        label = (
            tools._node_label(self._wrapper, include_name=True)
            if tools is not None
            else self.name
        )
        raise TypeError(
            f"{label} is a managed {self.kind} row, not an ImageTool. "
            "Use one of this row's ImageTool children instead."
        )

    def show(self) -> None:
        """Show the managed tool window."""
        self._wrapper.show()

    def hide(self) -> None:
        """Hide the managed tool window."""
        self._wrapper.hide()

    def dispose(self) -> None:
        """Dispose of the managed tool window."""
        self._wrapper.dispose()

    def __getattr__(self, attr: str) -> typing.Any:
        if hasattr(self._wrapper, attr):
            value = getattr(self._wrapper, attr)
            if callable(value):
                return value
        raise AttributeError(attr)

    def __getitem__(self, _key: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __add__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __radd__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __sub__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rsub__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __mul__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rmul__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __matmul__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rmatmul__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __truediv__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rtruediv__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __floordiv__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rfloordiv__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __mod__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rmod__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __pow__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __rpow__(self, _other: typing.Any) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __neg__(self) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __pos__(self) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __abs__(self) -> typing.NoReturn:
        self._raise_not_imagetool()

    def __repr__(self) -> str:
        tools = self._console_tools
        label = (
            tools._node_label(self._wrapper, include_name=True)
            if tools is not None
            else f"{self.kind}: {self.name}"
        )
        out = f"{label}\n"
        out += f"  UID: {self.uid}\n"
        out += f"  Added: {self._wrapper.added_time_display}\n"
        out += f"  Children: {len(self.children)}\n"
        return out


class _ToolChildren:
    def __init__(self, parent: ToolNamespace | _ManagedToolNamespace) -> None:
        self._parent = parent

    def _nodes(self) -> list[_ManagedWindowNode]:
        tools = self._parent._console_tools
        if tools is None:
            return []
        return tools._child_nodes(self._parent._wrapper)

    def _wrap_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> ToolNamespace | _ManagedToolNamespace:
        tools = self._parent._console_tools
        if tools is None:
            raise LookupError("Parent was destroyed")
        return tools._node_handle(node)

    def __getitem__(
        self, index: int | slice
    ) -> (
        ToolNamespace
        | _ManagedToolNamespace
        | list[ToolNamespace | _ManagedToolNamespace]
    ):
        tools = self._parent._console_tools
        if tools is None:
            raise LookupError("Parent was destroyed")
        nodes = self._nodes()
        if isinstance(index, slice):
            return [self._wrap_node(node) for node in nodes[index]]
        return self._wrap_node(nodes[index])

    def __iter__(self):
        tools = self._parent._console_tools
        if tools is None:
            return iter(())
        return (self._wrap_node(node) for node in self._nodes())

    def __len__(self) -> int:
        return len(self._nodes())

    def __repr__(self) -> str:
        tools = self._parent._console_tools
        if tools is None:
            return "No children"
        nodes = self._nodes()
        parent_label = tools._node_label(self._parent._wrapper, include_name=False)
        if not nodes:
            return f"No children for {parent_label}"
        lines = [f"Children for {parent_label}:"]
        self._append_tree_lines(lines, self._parent._wrapper, "")
        return "\n".join(lines)

    def _append_tree_lines(
        self,
        lines: list[str],
        parent: _ImageToolWrapper | _ManagedWindowNode,
        prefix: str,
    ) -> None:
        tools = self._parent._console_tools
        if tools is None:
            return
        nodes = tools._child_nodes(parent)
        for index, node in enumerate(nodes):
            is_last = index == len(nodes) - 1
            connector = "└─ " if is_last else "├─ "
            lines.append(f"{prefix}{connector}[{index}] {tools._tree_node_label(node)}")
            child_prefix = prefix + ("   " if is_last else "│  ")
            self._append_tree_lines(lines, node, child_prefix)


class _DerivedDataNamespace(_ConsoleDataHandleBase):
    def __init__(
        self,
        tools: ToolsNamespace | None,
        detached: xr.DataArray | Callable[[], xr.DataArray] | None,
        expression: str,
        script_inputs: Sequence[ScriptInput],
        *,
        copyable: bool,
        code_prelude: Sequence[str] = (),
        operations: Sequence[ToolProvenanceOperation] = (),
        seed_expression: str | None = None,
        bound_step: ReplayStep | None = None,
        xarray_options: dict[str, typing.Any] | None = None,
        previous: _ConsoleDataHandleBase | None = None,
    ) -> None:
        self._tools = tools
        self._expression = expression
        self._script_inputs = _dedupe_script_inputs(script_inputs)
        self._copyable = copyable
        self._code_prelude = _dedupe_code_preludes(code_prelude)
        self._operations = _kspace_conversion.stamp_kspace_conversion_groups(operations)
        self._seed_expression = seed_expression
        self._bound_step = bound_step
        self._console_name: str | None = None
        self._xarray_options = (
            _calculation_options() | (xarray_options or {})
            if tools is not None
            else xarray_options
        )
        self._captured_input: CalculatedInput | None = None
        if tools is None:
            if detached is None:
                raise ValueError("A detached console result needs its data")
            value = detached() if callable(detached) else detached
            self._data = capture_display_data(value, live=True)
        else:
            if not self._script_inputs or not copyable:
                _snapshot_required("A console calculation without recorded inputs")
            item = CalculatedInput(name="calculated", label="Console calculation")
            if (
                isinstance(previous, _DerivedDataNamespace)
                and previous._console_name is None
                and previous._operations
                and len(self._operations) == 1
                and len(self._script_inputs) == 1
                and isinstance(source := self._script_inputs[0], CalculatedInput)
                and source._capture.revision_id == previous._calculation_revision
            ):
                # The portable recipe presents a chain of typed steps together.
                # The new application takes only its immediate graph operand.
                self._operations = _kspace_conversion.stamp_kspace_conversion_groups(
                    (*previous._operations, *self._operations)
                )
                self._script_inputs = previous._script_inputs
                self._seed_expression = previous._seed_expression
                self._code_prelude = _dedupe_code_preludes(
                    previous._code_prelude, self._code_prelude
                )
                operation = self._operations[-1]
                marker = operation.group
                if marker is not None and marker.index == marker.size - 1:
                    prefix = tuple(
                        (getattr(member, "op", ""), member.group.focus)
                        for member in self._operations[-marker.size : -1]
                        if member.group is not None
                    )
                    operation = operation.model_copy(
                        update={
                            "group": marker.model_copy(
                                update={"console_prefix": prefix}
                            )
                        }
                    )
                operands: Sequence[ScriptInput] = (source,)
            else:
                operands = script_inputs
                if bound_step is not None:
                    operation = bound_step.operation
                elif self._operations:
                    # Each managed handle captures the result of its preceding call.
                    # The receiver therefore supplies the only input to this step.
                    (operation,) = self._operations
                else:
                    operation = RecordedProgram(
                        spec=self._expression_spec(
                            "derived", "Evaluate console expression", operands
                        )
                    )
            if not isinstance(operation, RecordedProgram):
                operation = operation.capture_execution_options()
                if self._bound_step is not None:
                    self._bound_step = self._bound_step.model_copy(
                        update={"operation": operation}
                    )
                elif self._operations:
                    self._operations = (*self._operations[:-1], operation)
            tools._manager._data_graph.calculate_console(
                operation, operands, item._capture
            )
            self._captured_input = item
            self._data = item._capture.calculated_data

    @property
    def data(self) -> xr.DataArray:
        return self._data.copy(deep=True)

    @property
    def _console_tools(self) -> ToolsNamespace | None:
        return self._tools

    @property
    def _calculation_revision(self) -> str | None:
        return (
            self._captured_input._capture.revision_id
            if self._captured_input is not None
            else None
        )

    def _set_console_name(self, name: str) -> None:
        if _replay_name_reserved(name):
            return
        self._console_name = name

    def _console_operand(self) -> _ConsoleOperand:
        if (
            self._captured_input is None
            and self._console_name is None
            and self._bound_step is None
            and self._xarray_options is None
        ):
            return _ConsoleOperand(
                self._data,
                _expression_receiver_code(self._expression),
                self._script_inputs,
                self._copyable,
                self._code_prelude,
                self._seed_expression,
                self._operations,
            )
        return self._named_operand(self._console_name or "calculated")

    def _named_operand(self, name: str) -> _ConsoleOperand:
        if self._captured_input is not None:
            item = self._captured_input
            if item.name != name:
                alias = item.model_copy(
                    update={"name": name, "label": f"console variable {name!r}"}
                )
                item = alias
            return _ConsoleOperand(self._data, name, (item,), self._copyable)
        provenance_spec = self._console_provenance_spec(
            active_name=name, label=f"Use calculated input {name!r}"
        )
        item = ScriptInput(
            name=name,
            label=f"console variable {name!r}",
            provenance_spec=provenance_spec.model_dump(mode="json")
            if provenance_spec
            else None,
        )
        retained = list(self._script_inputs)
        weakref.finalize(item, retained.clear).atexit = False
        return _ConsoleOperand(
            self._data,
            name,
            (item,),
            provenance_spec is not None,
        )

    def _console_provenance_spec(
        self, *, active_name: str, label: str
    ) -> ToolProvenanceSpec | None:
        tools = self._console_tools
        inputs = self._script_inputs
        if tools is not None and self._calculation_revision is not None:
            adapter = tools._manager._data_graph
            # Export nested histories only when a caller needs the portable form.
            inputs = tuple(
                item.model_copy(
                    update={
                        "provenance_spec": adapter.export_spec(
                            item._capture.revision_id
                        ).model_dump(mode="json")
                    }
                )
                if isinstance(item, CalculatedInput)
                else item
                for item in inputs
            )
            step = adapter.graph.recipe_for(self._calculation_revision)
            if isinstance(step, Recipe) and isinstance(step.operation, RecordedProgram):
                return step.operation.with_output_name(
                    active_name
                ).spec.with_replay_inputs(inputs)
        if not inputs:
            return None
        if self._bound_step is not None:
            return recipe(
                self._bound_step,
                start_label=label,
                active_name=active_name,
                script_inputs=inputs,
                primary_input=inputs[0].name,
            )
        if self._operations and self._seed_expression is not None:
            seed_code = _script_code(
                self._code_prelude,
                f"{active_name} = {self._seed_expression}",
            )
            return script(
                *self._operations,
                start_label="Run ImageTool manager console code",
                seed_code=seed_code,
                active_name=active_name,
                script_inputs=inputs,
            )
        return self._expression_spec(active_name, label, inputs)

    def _expression_spec(
        self, active_name: str, label: str, inputs: Sequence[ScriptInput]
    ) -> ToolProvenanceSpec:
        assignment = f"{active_name} = {self._expression}"
        if self._xarray_options:
            options = ", ".join(
                f"{key}={value!r}" for key, value in self._xarray_options.items()
            )
            assignment = f"with xr.set_options({options}):\n    {assignment}"
        code = _script_code(self._code_prelude, assignment)
        return script(
            ScriptCodeOperation(
                label=label,
                code=code,
                copyable=self._copyable,
            ),
            start_label="Run ImageTool manager console code",
            active_name=active_name,
            script_inputs=inputs,
        )

    def __repr__(self) -> str:
        return repr(self._data)


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

    def checkpoint(self, revision_id: str) -> xr.DataArray:
        """Return a copy of a recorded source array named by Copy Full Code."""
        graph = self._manager._data_graph.graph
        if not graph.revision(revision_id).checkpointed:
            raise ValueError(f"Revision {revision_id!r} is not a source checkpoint")
        return graph.data(revision_id)

    def apply(
        self, func: Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any
    ) -> _DerivedDataNamespace:
        """Calculate a supported library operation from console handles.

        The console's supplied library aliases use this interface for direct calls
        such as ``xr.concat([tools[0], tools[1]], dim="scan")``. Use it explicitly
        with functions imported separately. It supports registered ERLab
        transformations, multi-input ``era.transform.shift``, and ``xr.concat``.
        Arguments can contain handles and literal settings. Other functions and
        callbacks require ordinary snapshots obtained through ``.data``.

        Examples
        --------
        >>> tools.apply(xr.concat, [tools[0], tools[1]], dim="scan")
        >>> tools.apply(era.transform.rotate, tools[0], 5.0, reshape=False)

        .. versionadded:: 3.30.0
        """
        if isinstance(func, _ConsoleLibraryFunction):
            func = func._function
        owner = _first_console_handle((args, kwargs))
        if owner is None:
            _snapshot_required("tools.apply without a handle input")
        options = {}
        if func is xr.concat or func is erlab.analysis.transform.shift:
            bound = inspect.signature(func).bind(*args, **kwargs)
            arguments = dict(bound.arguments)
            handles = (
                arguments.pop("objs")
                if func is xr.concat
                else (arguments.pop("darr"), arguments.pop("shift"))
            )
            if (
                isinstance(handles, (tuple, list))
                and len(handles) >= 2
                and all(
                    isinstance(handle, _ConsoleDataHandleBase) for handle in handles
                )
            ):
                operation = _bound_library_operation(func, len(handles), arguments)
                if operation is not None:
                    captured: dict[int, _ConsoleOperand] = {}
                    for index, handle in enumerate(handles):
                        if id(handle) not in captured:
                            captured[id(handle)] = (
                                handle._named_operand(
                                    handle._console_name or f"operand_{index}"
                                )
                                if isinstance(handle, _DerivedDataNamespace)
                                else handle._console_operand()
                            )
                    operands, inputs, copyable, prelude = _merge_operands(
                        *(captured[id(handle)] for handle in handles)
                    )
                    primary, *others = operands
                    slots = operation.input_slots()
                    names = dict(
                        zip(slots, (item.code for item in others), strict=True)
                    )
                    return _DerivedDataNamespace(
                        self,
                        None,
                        operation.expression_code_with_inputs(primary.code, names),
                        inputs,
                        copyable=copyable,
                        code_prelude=prelude,
                        bound_step=ReplayStep(
                            operation=operation, input_bindings=names
                        ),
                    )
            call_code, inputs, _, _, copyable, prelude = _format_call_code(args, kwargs)
            current = xr.get_options()
            options = {
                key: current[key]
                for key in ("keep_attrs", "use_new_combine_kwarg_defaults")
                if key in current
            }
            function_code = "xr.concat" if func is xr.concat else "era.transform.shift"
            expression = f"{function_code}({call_code})"
            seed_expression, operations = None, ()
        else:
            module_name = getattr(func, "__module__", "")
            name = getattr(func, "__name__", "")
            if not module_name.startswith("erlab.") or not name.isidentifier():
                _snapshot_required("This tools.apply function")
            module = importlib.import_module(module_name)
            if getattr(module, name, None) is not func:
                _snapshot_required("This tools.apply function")
            signature = inspect.signature(func)
            supplied = signature.bind(*args, **kwargs)
            settings = dict(supplied.arguments)
            source_name = next(iter(signature.parameters))
            receiver = settings.pop(source_name, None)
            if (
                not isinstance(receiver, _ConsoleDataHandleBase)
                or _first_console_handle(settings) is not None
            ):
                _snapshot_required("tools.apply with these input bindings")
            source = receiver._console_operand()
            for key, value in settings.items():
                if (
                    isinstance(value, xr.Dataset)
                    and key == "modelresult"
                    and func is erlab.analysis.gold.correct_with_edge
                ):
                    if any(
                        str(var).endswith("modelfit_results") for var in value.data_vars
                    ):
                        _snapshot_required("A console call with serialized fit results")
                elif isinstance(value, os.PathLike) and key == "model_path":
                    settings[key] = os.fsdecode(value)
                else:
                    # Check literal arguments before running a library operation.
                    settings[key] = _operand_from_value(value).value
            for param in signature.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    settings.update(settings.pop(param.name, {}))
            inputs, copyable, prelude = (
                source.script_inputs,
                source.copyable,
                source.code_prelude,
            )
            call = ConsoleCall(
                func=func,
                kwargs=settings,
                receiver_data=source.value,
                display_code=f"{module_name}.{name}",
                has_extra_tracked_inputs=False,
            )
            operation = _structured_operation_from_call(call)
            if operation is None:
                _snapshot_required(f"{module_name}.{name} with these arguments")
            expression = operation.expression_code(source.code)
            seed_expression, operations = _structured_seed_and_operations(
                source, operation
            )
        return _DerivedDataNamespace(
            self,
            None,
            expression,
            inputs,
            copyable=copyable,
            xarray_options=options or None,
            code_prelude=prelude,
            seed_expression=seed_expression,
            operations=operations,
            previous=owner,
        )

    @property
    def selected_data(self) -> list[xr.DataArray]:
        """Current DataArrays from selected root or child ImageTools."""
        return [
            ToolNamespace(self._manager._node_for_target(target), self).data
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

    def _node_handle(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> ToolNamespace | _ManagedToolNamespace:
        if node.is_imagetool:
            return ToolNamespace(node, self)
        return _ManagedToolNamespace(node, self)

    def _node_path(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[int] | None:
        path = self._manager._tool_graph.node_path(node)
        return None if path is None else list(path)

    def _child_nodes(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> list[_ManagedWindowNode]:
        nodes: list[_ManagedWindowNode] = []
        for child_uid in node._childtool_indices:
            child = self._manager._tool_graph.nodes.get(child_uid)
            if child is None or child.parent_uid != node.uid:
                continue
            nodes.append(child)
        return nodes

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
        if node.is_imagetool:
            label = f"ImageTool {display_index}"
        else:
            label = f"{self._node_kind(node)} {display_index}"
        if include_name and node.name:
            label += f": {node.name}"
        return label

    def _node_kind(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        if node.is_imagetool:
            return "ImageTool"
        return node.type_badge_text or "tool"

    def _tree_node_label(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        if node.is_imagetool:
            return self._node_label(node, include_name=True)
        label = self._node_kind(node)
        if node.name:
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
        publication: _CalculatedResult,
        *,
        authored: bool = False,
        source_owners: Iterable[str] = (),
        **kwargs,
    ) -> bool:
        display_kwargs = dict(kwargs)
        display_kwargs.pop("execute", None)
        manager = self._manager
        with manager._workspace_controller.local_provenance_change(
            edited_entries=lambda: (
                _console_authored_entries(
                    manager,
                    node,
                    node.provenance_spec,
                )
                if authored
                else ()
            ),
            source_owners=lambda: {
                node.uid: manager._data_graph.graph.follows(node.uid)
                | set(source_owners)
            },
        ):
            index = manager.add_imagetool(
                publication,
                show=True,
                activate=True,
                _display_options=display_kwargs,
            )
            node = manager._node_for_target(index)
        return True

    def _show_handle(
        self,
        handle: _ConsoleDataHandleBase,
        *,
        active_name: str,
        label: str,
        **kwargs: typing.Any,
    ) -> bool:
        replace_target = kwargs.pop("replace", None)
        if replace_target is not None:
            if isinstance(replace_target, str):
                target = replace_target
            else:
                try:
                    target = operator.index(replace_target)
                except TypeError:
                    targets = list(replace_target)
                    if len(targets) != 1:
                        raise ValueError(
                            "One console handle requires one replacement target"
                        ) from None
                    target = operator.index(targets[0])
            if isinstance(target, int) and target < 0:
                target = sorted(self._manager._tool_graph.root_wrappers)[target]
            if target != self._manager.next_idx:
                node = self._manager._node_for_target(target)
                if not node.is_imagetool:
                    raise TypeError("The replacement target must be an ImageTool")
                ToolNamespace(node, self).data = handle
                self._manager._sigDataReplaced.emit()
                return True
            # Replacement accepts the next unused index as a new window. As in
            # the public itool API, display options are ignored for replacement.
            kwargs = {}
        publication: _CalculatedResult | None = None
        operand = handle._console_operand()
        authored = isinstance(handle, _DerivedDataNamespace)
        if isinstance(handle, ToolNamespace):
            if handle._wrapper.manager is not self._manager:
                _snapshot_required("Displaying a handle outside its Manager")
            publication = handle._view_publication(active_name, operand)
            source_owners = {handle.uid}
        else:
            if (
                handle._console_tools is not None
                and handle._console_tools._manager is not self._manager
            ):
                _snapshot_required("Displaying a calculation outside its Manager")
            if (
                isinstance(handle, _DerivedDataNamespace)
                and handle._captured_input is not None
            ):
                publication = _CalculatedResult(
                    handle._captured_input._capture, active_name
                )
                source_owners = set()
            else:
                _snapshot_required("Displaying an unrecorded result")
        return self._show_dataarray_with_provenance(
            publication,
            authored=authored,
            source_owners=source_owners,
            **kwargs,
        )

    def itool(
        self, data: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        if isinstance(data, _ConsoleDataHandleBase):
            if args or not self._manager_argument_targets_this_manager(
                kwargs.get("manager")
            ):
                _snapshot_required("Displaying a handle outside its Manager")
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
            _snapshot_required("Displaying this result as one ImageTool")
        if _first_console_handle(data) is not None:
            _snapshot_required("Displaying a collection of console handles")
        return erlab.interactive.itool(data, *args, **kwargs)

    def _qshow_handle(
        self,
        data: _ConsoleDataHandleBase,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        return self.itool(data, *args, **kwargs)

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
        self._kernel_shutdown_requested = False
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
        if self._kernel_shutdown_requested:
            return
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
        if self._kernel_shutdown_requested:
            return
        if not self.kernel_manager.kernel and not self._kernel_initializing:
            self.initialize_kernel()
        super().execute(source, hidden=hidden, interactive=interactive)

    def store_data_as(self, node_uid: str, name: str) -> None:
        """Store the data in an ImageTool with IPython to reuse in other scripts."""
        self.initialize_kernel()
        tools = self._tools_namespace
        if tools is None:  # pragma: no cover - manager consoles always inject tools
            return
        node = tools._manager._tool_graph.nodes.get(node_uid)
        if node is None or not node.is_imagetool:
            return
        path = tools._node_path(node)
        if not path:  # pragma: no cover - registered nodes always have tree paths
            return
        tool_expression = f"tools[{path[0]}]"
        tool_expression += "".join(f".children[{child_row}]" for child_row in path[1:])
        store_commands = (
            f"{name} = {tool_expression}.data",
            f"get_ipython().run_line_magic('store', '{name}')",
            f"del {name}",
        )
        self.execute("\n".join(store_commands), hidden=True)

    @QtCore.Slot()
    def shutdown_kernel(self) -> None:
        self._kernel_shutdown_requested = True
        if self.kernel_manager.kernel:
            shell = self.kernel_manager.kernel.shell
            history_thread = getattr(shell.history_manager, "save_thread", None)
            if self._tools_namespace is not None:
                self._tools_namespace.unbind_shell()
                self._tools_namespace = None
            if self._erlab_io_hooks_registered:
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
            atexit.unregister(shell.atexit_operations)
            if history_thread is not None:
                history_thread.stop()
            try:
                shell.atexit_operations()
            finally:
                shell.__class__.clear_instance()

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
            "\033[1mTip:\033[0m Drag a row here to insert its tools[...] "
            "expression.\n\n"
            + _command_ansi(
                "Access raw data",
                [
                    "tools[<index>].data",
                    "tools[<index>].children[0].children[0].data",
                    "tools.selected_data",
                ],
            )
            + "\n"
            + _command_ansi(
                "Track provenance",
                [
                    "tools[0] - tools[1]",
                    "tools[0].children[0].children[0] - tools[1]",
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

    def _initialize_visible_console(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self, self._console_widget):
            return
        if self._console_widget._kernel_shutdown_requested:
            return
        self._console_widget.initialize_kernel()
        self._console_widget._update_colors()

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if not hasattr(self, "_console_widget") or event is None:
            return False
        if obj is not self._console_widget:
            return super().eventFilter(obj, event)
        if not erlab.interactive.utils.qt_is_valid(self, self._console_widget, event):
            return False
        try:
            event_type = event.type()
        except RuntimeError:
            return False
        if event_type == QtCore.QEvent.Type.Show:
            erlab.interactive.utils.single_shot(
                self._console_widget,
                0,
                self._initialize_visible_console,
                self,
            )
        return super().eventFilter(obj, event)

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            self._console_widget._update_colors()

        super().changeEvent(evt)
